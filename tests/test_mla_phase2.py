import unittest

from serving.core.memory_model import (
    calculate_sizes,
    full_cluster_kv_bytes_per_token,
)
from serving.core.trace_generator import resolve_variant


class MLAPhase2Test(unittest.TestCase):
    def test_deepseek_mla_kv_and_variant(self):
        model = "deepseek-ai/DeepSeek-V3"
        _, _, output_size = calculate_sizes(
            model, "mla_attention", 1, kv_len=1024, parallel=8, fp=2,
        )
        fp16_input, _, _ = calculate_sizes(
            model, "mla_attention", 1, kv_len=1024, parallel=8, fp=2,
        )
        fp8_input, _, _ = calculate_sizes(
            model, "mla_attention", 1, kv_len=1024, parallel=8, fp=2,
            kv_cache_dtype="fp8",
        )

        self.assertEqual(output_size, 7168 * 2)
        self.assertEqual(fp16_input - fp8_input, (512 + 64) * 1024)
        self.assertEqual(
            full_cluster_kv_bytes_per_token(model, 16, tp_size=8),
            (512 + 64) * 61 * 8 * 2,
        )
        self.assertEqual(
            resolve_variant(None, "auto", {"torch_dtype": "bfloat16",
                                             "quantization_config": {"quant_method": "fp8"}}),
            "bf16-fp8",
        )


if __name__ == "__main__":
    unittest.main()
