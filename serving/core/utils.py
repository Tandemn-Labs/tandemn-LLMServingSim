import os
from time import time
import json

from .run_paths import input_path


# Formatting string for a trace file's per-layer row. Kept in this
# module because trace writers live across the codebase and import it
# as the canonical row template.
_FMT = (
    "{:<30}"  # Layername
    "{:<15}"  # comp_time
    "{:<15}"  # input_loc
    "{:<15}"  # input_size
    "{:<15}"  # weight_loc
    "{:<15}"  # weight_size
    "{:<15}"  # output_loc
    "{:<15}"  # output_size
    "{:<15}"  # comm_type
    "{:<15}"  # comm_size
    "{:<15}"  # misc
    "\n"
)


def get_workload(batch, hardware, instance_id=0, event=False, workload_name=None, inputs_root=None):
    if event:
        file_name = 'event_handler'
    elif workload_name:
        file_name = workload_name
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    return input_path(inputs_root, "workload", file_name, "llm")


def header():
    string_list = [
        "Layername", "comp_time", "input_loc", "input_size",
        "weight_loc", "weight_size", "output_loc", "output_size",
        "comm_type", "comm_size", "misc",
    ]
    ileft_list = [30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    output = ""
    for string, ileft in zip(string_list, ileft_list):
        output += ('{0:<' + str(ileft) + '}').format(string)
    output += '\n'
    return output


def formatter(layername, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc):
    return _FMT.format(
        layername, comp_time, input_loc, input_size, weight_loc,
        weight_size, output_loc, output_size, comm_type, comm_size, misc,
    )


def get_config(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(base_dir)
    repo_root = os.path.dirname(serving_dir)
    candidate_paths = [
        os.path.join(repo_root, "configs", "model", model_name + ".json"),
        os.path.join(serving_dir, "configs", "model", model_name + ".json"),
    ]

    config = None
    for config_path in candidate_paths:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            break
        except FileNotFoundError:
            continue

    if config is None:
        raise FileNotFoundError(
            f"Config file for model '{model_name}' not found. Checked: "
            f"{', '.join(candidate_paths)}. Please add the corresponding config file."
        )

    return config


def get_num_routed_experts(config):
    """Return the model's routed expert count, or zero for dense models."""
    for key in ("num_local_experts", "num_experts", "n_routed_experts"):
        if key in config and config[key] is not None:
            return int(config[key])
    return 0


def qwen3_next_layer_type(config, layer_num):
    """Return the configured Qwen3-Next attention type for a layer."""
    layer_types = config.get("layer_types")
    if layer_types:
        return layer_types[int(layer_num)]
    interval = max(1, int(config.get("full_attention_interval", 4)))
    return (
        "full_attention"
        if (int(layer_num) + 1) % interval == 0
        else "linear_attention"
    )


def is_encoder_model(config):
    """Return whether the config selects a supported encoder-only model."""
    architectures = config.get("architectures") or []
    return any(
        arch in ("XLMRobertaModel", "XLMRobertaForSequenceClassification")
        for arch in architectures
    )


def encoder_task(config):
    architectures = config.get("architectures") or []
    if "XLMRobertaForSequenceClassification" in architectures:
        return "rerank"
    if "XLMRobertaModel" in architectures:
        return "embed"
    return "generate"


def effective_max_model_len(config):
    """Usable token limit, accounting for XLM-R's two position offsets."""
    limit = int(config["max_position_embeddings"])
    return limit - 2 if is_encoder_model(config) else limit


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    config = get_config(model_name)

    if config:
        print(f"Loaded config for {model_name}: {list(config.keys())[:5]}")
        print(config['model_type'])
