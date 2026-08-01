import os, threading
from .utils import (
    encoder_task,
    get_config,
    get_num_routed_experts,
    is_encoder_model,
    qwen3_next_layer_type,
)
from .radix_tree import *
import logging
from enum import Enum

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024
_MLA_MODEL_TYPES = frozenset(("deepseek_v3", "kimi_k2"))


def _is_fp8_checkpoint(config):
    quantization = config.get("quantization_config")
    return (
        isinstance(quantization, dict)
        and str(quantization.get("quant_method", "")).lower() == "fp8"
    )


def _linear_weight_size(config, rows, cols, fp):
    """Return serialized bytes for one linear weight matrix."""
    rows, cols = int(rows), int(cols)
    if not _is_fp8_checkpoint(config):
        return rows * cols * fp

    block = config.get("quantization_config", {}).get("weight_block_size")
    if isinstance(block, (list, tuple)) and len(block) == 2:
        block_rows, block_cols = max(int(block[0]), 1), max(int(block[1]), 1)
        scales = (
            (rows + block_rows - 1) // block_rows
            * ((cols + block_cols - 1) // block_cols)
            * 4
        )
    else:
        scales = 4
    return rows * cols + scales


def _encoder_head_fp(config, model_fp):
    """Bytes per element for vLLM pooling heads.

    vLLM defaults pooling heads to FP32 even when encoder blocks are FP16.
    A checkpoint can opt back into the model dtype with ``head_dtype=model``.
    """
    head_dtype = str(config.get("head_dtype", "float32")).lower()
    if head_dtype in ("model", "float16", "half", "fp16", "bfloat16", "bf16"):
        return model_fp
    return 4

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3

class MemoryModel():
    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto'):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage

        self.config = get_config(model)
        self.model_type = self.config.get("model_type")
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        self.is_moe = get_num_routed_experts(self.config) > 0
        self.is_encoder = is_encoder_model(self.config)
        self.is_qwen3_next = self.model_type == "qwen3_next"
        self.first_k_dense_replace = int(
            self.config.get("first_k_dense_replace", 0)
        )

        if self.is_qwen3_next and enable_prefix_caching:
            raise ValueError(
                "qwen3_next prefix caching is not supported yet because "
                "DeltaNet state is not prefix-addressable; use "
                "--no-enable-prefix-caching"
            )

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.npu_used = self.weight
        self.cpu_used = 0
        if self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
                    # print("[instance {}] prefix_cache_capacity : {}".format(instance_id, prefix_cache_capacity // GB_TO_BYTE))
                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.num_npus),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache
    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.fp
        weight = 0

        if self.is_encoder:
            _, embedding, _ = calculate_sizes(
                self.model, "embedding", 1, parallel=tp, fp=fp,
            )
            weight = embedding + self._get_encoder_block_weight(tp, fp) * self.n_layer
            head = (
                "classifier_head"
                if encoder_task(self.config) == "rerank"
                else "mean_pool"
            )
            _, head_weight, _ = calculate_sizes(
                self.model, head, 1, parallel=tp, fp=fp, sequences=1,
            )
            weight += head_weight
            self.logger.info("NPU: model weight %dMB loaded", weight // MB_TO_BYTE)
            return weight

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp)
        weight += embedding
        if self.is_qwen3_next:
            block_weights = [
                self._get_qwen3_next_block_weight(tp, ep, fp, layer_num)
                for layer_num in range(self.n_layer)
            ]
            weight += self._max_pp_stage_weight(block_weights, pp)
        elif self.first_k_dense_replace > 0:
            block_weights = [
                self._get_weight_per_block(
                    tp, ep, fp,
                    is_moe_block=layer_num >= self.first_k_dense_replace,
                )
                for layer_num in range(self.n_layer)
            ]
            weight += self._max_pp_stage_weight(block_weights, pp)
        else:
            weight += self._get_weight_per_block(tp, ep, fp) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_encoder_block_weight(self, tp, fp):
        return sum(
            calculate_sizes(self.model, layer, 1, parallel=tp, fp=fp)[1]
            for layer in (
                "qkv_proj", "o_proj", "attn_layernorm", "intermediate",
                "output", "output_layernorm",
            )
        )

    @staticmethod
    def _max_pp_stage_weight(block_weights, pp):
        if pp <= 1:
            return sum(block_weights)
        return max(
            sum(block_weights[stage * len(block_weights) // pp:
                              (stage + 1) * len(block_weights) // pp])
            for stage in range(pp)
        )

    def _max_pp_stage_layer_count(self, layer_type):
        pp = max(self.pp_size, 1)
        return max(
            sum(
                qwen3_next_layer_type(self.config, layer_num) == layer_type
                for layer_num in range(
                    stage * self.n_layer // pp,
                    (stage + 1) * self.n_layer // pp,
                )
            )
            for stage in range(pp)
        )

    def _get_weight_per_block(self, tp, ep, fp, is_moe_block=None):
        """Per-block weight: dense layers use TP, MoE experts use EP."""
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp)
        block_weight += ln_w  # input layernorm
        if self.model_type in _MLA_MODEL_TYPES:
            _, attn_w, _ = calculate_sizes(
                self.model, "mla_attention", 1, parallel=tp, fp=fp,
            )
            block_weight += attn_w
        else:
            _, qkv_w, _ = calculate_sizes(
                self.model, "qkv_proj", 1, parallel=tp, fp=fp,
            )
            _, o_w, _ = calculate_sizes(
                self.model, "o_proj", 1, parallel=tp, fp=fp,
            )
            block_weight += qkv_w + o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if is_moe_block is None:
            is_moe_block = self.is_moe
        if is_moe_block:
            _, moe_w, _ = calculate_sizes(
                self.model, 'moe', 1, parallel=ep, tp_parallel=tp, fp=fp,
            )
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn2_w
        return block_weight

    def _get_qwen3_next_block_weight(self, tp, ep, fp, layer_num):
        attention_layer = (
            "full_attention"
            if qwen3_next_layer_type(self.config, layer_num) == "full_attention"
            else "delta_net"
        )
        _, attn_w, _ = calculate_sizes(
            self.model, attention_layer, 1, parallel=tp, fp=fp,
        )
        _, ln_w, _ = calculate_sizes(
            self.model, "layernorm", 1, parallel=tp, fp=fp,
        )
        _, moe_w, _ = calculate_sizes(
            self.model, "moe", 1, parallel=ep, tp_parallel=tp, fp=fp,
        )
        return attn_w + 2 * ln_w + moe_w

    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        if self.is_encoder:
            return 0
        if self.model_type in _MLA_MODEL_TYPES:
            pp = max(self.pp_size, 1)
            local_layers = max(
                (stage + 1) * self.n_layer // pp - stage * self.n_layer // pp
                for stage in range(pp)
            )
            latent_dim = (
                int(self.config["kv_lora_rank"])
                + int(self.config["qk_rope_head_dim"])
            )
            # MLA stores one compressed KV latent per layer and replicates it
            # across TP ranks, unlike ordinary MHA/GQA cache sharding.
            return latent_dim * seq * local_layers * self.kv_fp
        if self.is_qwen3_next:
            full_layers = self._max_pp_stage_layer_count("full_attention")
            local_kv_heads = max(1, self.kv_head // max(self.tp_size, 1))
            return 2 * local_kv_heads * self.head_dim * seq * full_layers * self.kv_fp

        # K & V multiply 2
        return 2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus

    def get_encoder_activation(self, requests):
        """Conservative peak transient tensor bytes for an encoder batch."""
        if not self.is_encoder:
            return 0
        tokens = sum(req.original_input for req in requests)
        hidden = tokens * self.n_embd * self.fp
        qkv = 3 * hidden
        ffn = tokens * self.config["intermediate_size"] * self.fp
        return hidden + max(qkv, ffn)

    def _qwen3_next_state_bytes(self):
        if not self.is_qwen3_next:
            return 0
        linear_layers = self._max_pp_stage_layer_count("linear_attention")
        tp = max(self.tp_size, 1)
        channels = (
            2 * self.config["linear_num_key_heads"] * self.config["linear_key_head_dim"]
            + self.config["linear_num_value_heads"] * self.config["linear_value_head_dim"]
        )
        conv_state = (
            (self.config["linear_conv_kernel_dim"] - 1)
            * channels // tp
        )
        ssm_state = (
            self.config["linear_num_value_heads"] // tp
            * self.config["linear_value_head_dim"]
            * self.config["linear_key_head_dim"]
        )
        return linear_layers * (conv_state + ssm_state) * self.fp
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        # ceil division: (n + block_size - 1) // block_size
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        total = self.get_kv(num_blocks * self.block_size)
        if self.is_qwen3_next and req.num_computed_tokens > 0:
            total += self._qwen3_next_state_bytes()
        return total

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    # scheduled_tokens: dict mapping request id to number of tokens scheduled this step
    # 
    # vLLM-style cumulative allocation:
    #   blocks_after = ceil((computed + scheduled) / block_size)
    #   blocks_before = ceil(computed / block_size) if computed > 0 else 0
    #   new_blocks = blocks_after - blocks_before
    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        # print("[get_block_kv] current batch_req length : {}".format(batch_len))
        block_kv_size = 0
        for i in range(batch_len):
            req = batch_req[i]
            if req.evict or req.is_prefill():
                # Prefill and reloaded decode requests may allocate newly
                # computed blocks. Existing evicted KV is reloaded separately
                # by Scheduler.load_size.
                hit = req.npu_cache_hit if self.enable_prefix_caching else 0
                
                if scheduled_tokens and req.id in scheduled_tokens:
                    tokens_this_step = scheduled_tokens[req.id]
                else:
                    raise RuntimeError("[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: scheduled_tokens cannot be None")
                
                # vLLM-style cumulative block allocation
                computed_before = req.num_computed_tokens
                
                total_after = computed_before + tokens_this_step
                
                # Calculate blocks needed (cumulative)
                blocks_after = (total_after + self.block_size - 1) // self.block_size
                blocks_before = (computed_before + self.block_size - 1) // self.block_size if computed_before > 0 else 0
                
                
                new_blocks = max(0, blocks_after - blocks_before)
                block_kv_size += self.get_kv(new_blocks * self.block_size)
                if self.is_qwen3_next and computed_before == 0 and new_blocks > 0:
                    block_kv_size += self._qwen3_next_state_bytes()
                # print("[DEBUG] hit : {} | tokens_this_step : {} | computed_before : {} | total_after : {} | new_blocks : {} | block_kv_size : {}".format(
                #     hit, tokens_this_step, computed_before, total_after, new_blocks, block_kv_size
                # ))
            else:
                # Decode: use num_computed_tokens (or input for backwards compat)
                computed = req.num_computed_tokens
                num_before = (computed + self.block_size - 1) // self.block_size if computed > 0 else 0
                num_after = (computed + 1 + self.block_size - 1) // self.block_size
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def get_evict_kv(self, req):
        evict_size = 0
        # Use num_computed_tokens if available, fallback to input for backwards compat
        computed = req.num_computed_tokens
        hit = req.npu_cache_hit if self.enable_prefix_caching else 0
        needed = max(0, computed - hit)
        # ceil division: (needed + block_size - 1) // block_size
        num_blocks = (needed + self.block_size - 1) // self.block_size
        evict_size += self.get_kv(num_blocks * self.block_size)
        if self.is_qwen3_next and needed > 0:
            evict_size += self._qwen3_next_state_bytes()
        return evict_size

    def free_weight(self):
        if self.npu_used - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.weight

    def is_free(self):
        is_free = self.npu_used == 0 and self.cpu_used == 0
        if not is_free:
            self.logger.error(
                "Memory leak detected: NPU used: %.2fMB, CPU used: %.2fMB",
                self.node_id,
                self.instance_id,
                self.npu_used / MB_TO_BYTE,
                self.cpu_used / MB_TO_BYTE,
            )
        return

    # -------------------- Memory Management --------------------
    
    def allocate(self, size, device):
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.avail_size()
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.avail_size()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    def evictable_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device): 
        # Increment lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
        elif (device == Device.CPU or device == Device.CXL) and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        # Decrement lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
            req.npu_last_node = None
            req._prefix_locked = False
        elif device == Device.CPU and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    def cache_unfinished_req(self, req, device):
        # Get new_last_node via cache_unfinished_req (replaces last node)
        # Decrement old node's lock ref count, increment new node's lock ref count
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            
            old_node = req.npu_last_node
            # print(f"[CACHE_UNFINISHED] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            if old_node is not None and req._prefix_locked:
                self.npu_prefix_cache.dec_lock_ref(old_node)
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            # print(f"[CACHE_UNFINISHED] req={req.id} AFTER: old_node_id={old_node.id}(lock_ref={old_node.lock_ref}) new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            req.npu_last_node = new_last_node
            req._prefix_locked = True
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            self.npu_prefix_cache.cache_finished_req(req)
            # Only dec_lock_ref if the request was locked
            node = req.npu_last_node
            if not req._prefix_locked:
                # Never locked → skip dec
                pass
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref={node.lock_ref if node else 'N/A'} (SKIPPED dec - not locked)")
            else:
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                if node is not None:
                    self.npu_prefix_cache.dec_lock_ref(node)
                    req.npu_last_node = None
                req._prefix_locked = False
            # node = req.npu_last_node
            # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
            # self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_AFTER={node.lock_ref if node else 'N/A'}")
            # print(f"[CACHE_FINISHED] req={req.id} evictable_size={self.npu_prefix_cache.evictable_size()} protected_size={self.npu_prefix_cache.protected_size()} total_size={self.npu_prefix_cache.total_size()}")
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============NPU PREFIX CACHE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_finished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def evict_prefix_cache(self, bytes, device):
        if not self.enable_prefix_caching or bytes <= 0:
            return

        if device == Device.NPU:
            cache = self.npu_prefix_cache
        elif device == Device.CPU:
            cache = self.second_tier_prefix_cache
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        # Each cache instance carries its own bytes-per-token in kv_size:
        # per-rank for NPU, full-cluster for the second-tier pool.
        space_needed = (bytes + cache.kv_size - 1) // cache.kv_size
        cache.evict(space_needed)

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req): # req.prefix_cache_hit initialization 
        if not self.enable_prefix_caching:
            return
        
        tokens = req.input_hash_ids
        if tokens is None:
            return
        old_node = req.npu_last_node
        res = self.npu_prefix_cache.match_prefix(tokens[:req.input])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node
        # print(f"[PREFIX_MATCH] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={res.last_device_node.id}(lock_ref={res.last_device_node.lock_ref}) hit={res.hit_length} num_computed={req.num_computed_tokens}")

        if self.prefix_storage is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:req.input])
            req.storage_cache_hit = res_storage.hit_length
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = 0
            req.storage_last_node = None
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)
        # if req.num_computed_tokens < req.prefix_cache_hit:
        #     req.num_computed_tokens = req.prefix_cache_hit
        if req.num_computed_tokens == 0:
            req.num_computed_tokens = req.prefix_cache_hit
            # print(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # for debugging
        
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    def erase_prefix_info(self, req):
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, self.prefix_storage)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        # if not self.enable_prefix_caching:
        #     return
        npu_byte_alloc = 0
        npu_byte_free = 0
        cpu_byte_alloc = 0
        cpu_byte_free = 0
        # self.npu_prefix_cache.take_events() -> [BlockStored, BlockStored, BlockRemoved, ...]
        for ev in self.npu_prefix_cache.take_events():
            # print(f" current event block: {ev}")
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                for h in ev.block_hashes:
                    # self._npu_cache_hashtolen[h] = tlen
                    if h in self._npu_cache_hashtolen:
                        self._npu_cache_hashtolen[h][1] += 1
                        # if self._npu_cache_hashtolen[h][1] >= 2:
                        #     print("duplicated hash occurs!! h : {}".format(h))
                    else:
                        self._npu_cache_hashtolen[h] = [tlen, 1]
                npu_byte_alloc += self.get_kv(tlen)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    # tlen = self._npu_cache_hashtolen.pop(h, 0)
                    # if tlen == 0:
                    if h in self._npu_cache_hashtolen:
                        tlen = self._npu_cache_hashtolen[h][0]
                        self._npu_cache_hashtolen[h][1] -= 1
                        if self._npu_cache_hashtolen[h][1] <= 0:
                            del self._npu_cache_hashtolen[h]
                        npu_byte_free += self.get_kv(tlen)
                    else:
                        print(f"[HASH_MISS] BlockRemoved hash={h} NOT FOUND in map (map_size={len(self._npu_cache_hashtolen)})")
                        self.logger.warning(f"NPU prefix cache remove unknown block hash {h}")
                    # else:
                    #     print(f"[HASH_HIT] BlockRemoved hash={h} tlen={tlen}")
                    # npu_byte_free += self.get_kv(tlen)
        # free first, then allocate
        if npu_byte_free > 0:
            self.free(npu_byte_free, Device.NPU)
        if npu_byte_alloc > 0:
            self.allocate(npu_byte_alloc, Device.NPU)
        # if npu_byte_free > 0:
        #     self.free(npu_byte_free, Device.NPU)

        # Second-tier (CPU/CXL) prefix cache events.
        if self.prefix_storage is None:
            return

        if self.prefix_storage is Device.CPU and not self.enable_prefix_sharing:
            # Non-shared CPU second_tier: bridge events into the instance's
            # cpu_used counter so allocations are bounded by cpu_mem.
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            self._cpu_cache_hashtolen[h][1] += 1
                        else:
                            self._cpu_cache_hashtolen[h] = [tlen, 1]
                    cpu_byte_alloc += self.get_kv(tlen) * self.num_npus
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            tlen = self._cpu_cache_hashtolen[h][0]
                            self._cpu_cache_hashtolen[h][1] -= 1
                            if self._cpu_cache_hashtolen[h][1] <= 0:
                                del self._cpu_cache_hashtolen[h]
                            cpu_byte_free += self.get_kv(tlen) * self.num_npus
                        else:
                            self.logger.warning(f"CPU prefix cache remove unknown block hash {h}")

            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)
            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
        else:
            # Shared pool or CXL: the cache itself accounts via
            # total_memory_usage (= kv_stored + total_size * kv_size),
            # so no instance-side counter update is needed. Drain the
            # queue to prevent it from growing unboundedly.
            self.second_tier_prefix_cache.take_events()

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.prefix_storage is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto', tp_size=1):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    if is_encoder_model(config):
        return 0
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)
    kv_dim = kv_head * head_dim
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    if config.get("model_type") in _MLA_MODEL_TYPES:
        latent_dim = (
            int(config["kv_lora_rank"])
            + int(config["qk_rope_head_dim"])
        )
        # MLA's compressed cache is replicated across TP ranks.
        return latent_dim * n_layer * max(int(tp_size), 1) * kv_fp
    # 2 (K + V) * kv_dim * n_layer * bytes_per_elem
    return 2 * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(
    model, layer_name, length, kv_len=None, pim=False, parallel=1,
    tp_parallel=None, fp=2, sequences=None, kv_cache_dtype='auto',
):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    q_dim = n_head * head_dim       # total Q projection output dim
    kv_dim = kv_head * head_dim     # total KV projection output dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    num_local_experts = get_num_routed_experts(config)
    is_encoder = is_encoder_model(config)
    encoder_head_fp = _encoder_head_fp(config, fp) if is_encoder else fp
    num_sequences = max(int(sequences if sequences is not None else length), 1)

    p = max(int(parallel), 1)
    tp = max(int(tp_parallel if tp_parallel is not None else p), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        if is_encoder:
            weight_size += (
                config["max_position_embeddings"] * n_embd
                + config.get("type_vocab_size", 1) * n_embd
                + 2 * n_embd
            ) * fp
        output_size = length * n_embd * fp

    elif layer_name in [
        "input_layernorm", "post_layernorm", "final_layernorm", "layernorm",
        "attn_layernorm", "output_layernorm",
    ]:
        input_size = length * n_embd * fp
        weight_size = (2 if is_encoder else 1) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp

    elif layer_name == "attention":
        if is_encoder:
            input_size = 3 * length * n_embd * fp // p
            weight_size = 0
            output_size = length * n_embd * fp // p
        elif not pim:
            input_size = (
                (n_head // p) * length * head_dim * fp +
                (kv_head // p) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * length * head_dim * fp
        else:
            input_size = (
                (n_head // p) * 1 * head_dim * fp +
                (kv_head // p) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * 1 * head_dim * fp

    elif layer_name == "mla_attention":
        q_lora_rank = int(config.get("q_lora_rank") or 0)
        kv_lora_rank = int(config["kv_lora_rank"])
        qk_nope_head_dim = int(config["qk_nope_head_dim"])
        qk_rope_head_dim = int(config["qk_rope_head_dim"])
        v_head_dim = int(config["v_head_dim"])
        local_heads = max(1, n_head // p)
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_fp = 1 if kv_cache_dtype == 'fp8' else fp

        input_size = (
            length * n_embd * fp
            + (kv_lora_rank + qk_rope_head_dim)
            * max(kv_len or 0, 0)
            * kv_fp
        )
        weight_size = (
            _linear_weight_size(config, n_embd, q_lora_rank, fp)
            + _linear_weight_size(config, q_lora_rank, local_heads * qk_head_dim, fp)
            + _linear_weight_size(
                config, n_embd, kv_lora_rank + qk_rope_head_dim, fp,
            )
            + _linear_weight_size(
                config,
                kv_lora_rank,
                local_heads * (qk_nope_head_dim + v_head_dim),
                fp,
            )
            + _linear_weight_size(config, local_heads * v_head_dim, n_embd, fp)
            + (q_lora_rank + kv_lora_rank) * fp
        )
        output_size = length * n_embd * fp

    elif layer_name == "full_attention":
        local_q_dim = q_dim // p
        local_kv_heads = max(1, kv_head // p)
        local_kv_dim = local_kv_heads * head_dim
        qkv_output_dim = 2 * local_q_dim + 2 * local_kv_dim
        input_size = (
            length * n_embd * fp
            + local_kv_dim * max(kv_len or 0, 0) * fp * 2
        )
        weight_size = (
            n_embd * qkv_output_dim
            + local_q_dim * n_embd
            + 2 * head_dim
        ) * fp
        output_size = length * n_embd * fp

    elif layer_name == "delta_net":
        linear_key_heads = config["linear_num_key_heads"]
        linear_value_heads = config["linear_num_value_heads"]
        key_head_dim = config["linear_key_head_dim"]
        value_head_dim = config["linear_value_head_dim"]
        channels = 2 * linear_key_heads * key_head_dim + linear_value_heads * value_head_dim
        weight_size = (
            n_embd * (2 * linear_key_heads * key_head_dim + 2 * linear_value_heads * value_head_dim) // p
            + n_embd * (2 * linear_value_heads) // p
            + config["linear_conv_kernel_dim"] * channels // p
            + (linear_value_heads * value_head_dim // p) * n_embd
            + value_head_dim
        ) * fp
        input_size = length * n_embd * fp
        output_size = length * n_embd * fp

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        qkv_dim = (q_dim + 2 * kv_dim) // p
        weight_size = _linear_weight_size(config, n_embd, qkv_dim, fp)
        if config.get("attention_bias", False):
            weight_size += qkv_dim * fp
        elif is_encoder:
            weight_size += qkv_dim * fp
        output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        input_size = length * (q_dim // p) * fp
        weight_size = _linear_weight_size(config, q_dim // p, n_embd, fp)
        if is_encoder:
            weight_size += n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "intermediate":
        input_size = length * n_embd * fp
        weight_size = (n_embd * (ffn_dim // p) + ffn_dim // p) * fp
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "activation":
        input_size = length * (ffn_dim // p) * fp
        weight_size = 0
        output_size = input_size

    elif layer_name == "output":
        input_size = length * (ffn_dim // p) * fp
        weight_size = ((ffn_dim // p) * n_embd + n_embd) * fp
        output_size = length * n_embd * fp

    elif layer_name == "mean_pool":
        input_size = length * n_embd * fp
        weight_size = 0
        output_size = num_sequences * n_embd * encoder_head_fp

    elif layer_name == "pooler_normalize":
        input_size = num_sequences * n_embd * encoder_head_fp
        weight_size = 0
        output_size = input_size

    elif layer_name == "classifier_head":
        # ClassifierPoolerHead converts the CLS vector to vLLM's FP32
        # pooling-head dtype before running RobertaClassificationHead.
        input_size = num_sequences * n_embd * fp
        num_labels = int(config.get("num_labels", len(config.get("id2label", {0: 0}))))
        weight_size = (
            n_embd * n_embd + n_embd
            + n_embd * num_labels + num_labels
        ) * encoder_head_fp
        output_size = num_sequences * num_labels * encoder_head_fp

    elif layer_name == "cls_pool":
        input_size = length * n_embd * fp
        weight_size = 0
        output_size = num_sequences * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = _linear_weight_size(config, n_embd, 2 * (ffn_dim // p), fp)
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = _linear_weight_size(config, ffn_dim // p, n_embd, fp)
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        routed_weight_size = (
            _linear_weight_size(config, n_embd, 2 * moe_ffn_dim, fp)
            + _linear_weight_size(config, moe_ffn_dim, n_embd, fp)
        )
        if config.get("model_type") in _MLA_MODEL_TYPES:
            shared_experts = int(config.get("n_shared_experts", 0) or 0)
            shared_size = shared_experts * moe_ffn_dim
            shared_weight_size = 0
            if shared_size:
                shared_weight_size = (
                    _linear_weight_size(config, n_embd, 2 * (shared_size // tp), fp)
                    + _linear_weight_size(config, shared_size // tp, n_embd, fp)
                )
            weight_size = (
                n_embd * num_local_experts * fp
                + num_local_experts * 4  # DeepSeek correction bias is FP32.
                + experts_per_rank * routed_weight_size
                + shared_weight_size
            )
        elif config.get("model_type") == "glm4_moe":
            shared_experts = int(config.get("n_shared_experts", 0) or 0)
            # GLM's router and correction bias are FP32; the routed experts
            # follow EP while shared experts are replicated across EP.
            weight_size = (
                (n_embd * num_local_experts + num_local_experts) * 4
                + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp
                + shared_experts * 3 * n_embd * moe_ffn_dim * fp // tp
            )
        elif config.get("model_type") == "qwen3_next":
            shared_size = int(config.get("shared_expert_intermediate_size", 0) or 0)
            weight_size = (
                n_embd * num_local_experts * fp
                + n_embd * fp  # shared expert gate
                + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp
                + 3 * n_embd * shared_size * fp // tp
            )
        else:
            weight_size = (
                n_embd * num_local_experts * fp  # gate (replicated)
                + experts_per_rank * routed_weight_size
            )
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        # FP8 checkpoint quantization does not apply to ParallelLMHead.
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
