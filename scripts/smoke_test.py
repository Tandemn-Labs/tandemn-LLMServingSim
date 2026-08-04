#!/usr/bin/env python3
"""Run one small simulation for every requested profiled hardware/model/TP case."""

import argparse
import csv
import json
import os
import re
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "microsoft/phi-4",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "zai-org/GLM-4.5-Air",
    "moonshotai/Kimi-K2-Instruct",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-reranker-v2-m3",
    "deepseek-ai/DeepSeek-V3",
}

# Edit these values when a profile represents a different SKU or interconnect.
HARDWARE = {
    "A10G": {"npu_mem_gb": 24, "npu_mem_bw_gbps": 600, "link_bw_gbps": 32,
              "link_latency_ns": 20_000, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "A100-40GB": {"npu_mem_gb": 40, "npu_mem_bw_gbps": 1555, "link_bw_gbps": 600,
                   "link_latency_ns": 0, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "A100-80GB": {"npu_mem_gb": 80, "npu_mem_bw_gbps": 2039, "link_bw_gbps": 600,
                   "link_latency_ns": 0, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "H100": {"npu_mem_gb": 80, "npu_mem_bw_gbps": 3350, "link_bw_gbps": 900,
             "link_latency_ns": 0, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "L4": {"npu_mem_gb": 24, "npu_mem_bw_gbps": 300, "link_bw_gbps": 32,
           "link_latency_ns": 20_000, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "L40S": {"npu_mem_gb": 48, "npu_mem_bw_gbps": 864, "link_bw_gbps": 64,
              "link_latency_ns": 20_000, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
    "RTXPRO6000-RUNPOD": {"npu_mem_gb": 96, "npu_mem_bw_gbps": 1597, "link_bw_gbps": 64,
                           "link_latency_ns": 20_000, "max_num_seqs": 256, "max_num_batched_tokens": 2048},
}


def read_yaml(path):
    with path.open() as f:
        return yaml.safe_load(f) or {}


def csv_layers(path):
    if not path.is_file():
        return set()
    with path.open(newline="") as f:
        return {row["layer"] for row in csv.DictReader(f)}


def profile_cases(root):
    for meta_path in sorted((root / "profiler/perf").glob("*/*/*/*/meta.yaml")):
        hardware, org, name, variant, _ = meta_path.relative_to(root / "profiler/perf").parts
        model = f"{org}/{name}"
        if model not in MODELS or hardware not in HARDWARE:
            continue
        for tp_dir in sorted(meta_path.parent.glob("tp*")):
            if tp_dir.is_dir() and tp_dir.name[2:].isdigit():
                yield {
                    "hardware": hardware,
                    "model": model,
                    "variant": variant,
                    "tp": int(tp_dir.name[2:]),
                    "bundle": meta_path.parent,
                }


def requirements(root, model):
    config = json.loads((root / "configs/model" / f"{model}.json").read_text())
    architecture = read_yaml(root / "profiler/models" / f"{config['model_type']}.yaml")
    sequence = architecture.get("sequence")
    if sequence is None:
        name = (config.get("architectures") or [None])[0]
        sequence = architecture["sequence_by_architecture"][name]
    needed = {category: set() for category in architecture["catalog"]}
    for section in sequence.values():
        layers = section.values() if isinstance(section, dict) else [section]
        for group in layers:
            for layer in group:
                for category, catalog in architecture["catalog"].items():
                    if layer in catalog:
                        needed[category].add(layer)
    return architecture, needed


def profile_gaps(root, case):
    architecture, needed = requirements(root, case["model"])
    gaps = []
    tp_dir = case["bundle"] / f"tp{case['tp']}"
    for category, layers in needed.items():
        filename = f"{category}.csv"
        if category in {"attention", "linear_attention"}:
            if not (tp_dir / filename).is_file():
                gaps.append(filename)
            continue
        if category == "moe":
            moe_dir = case["bundle"] / "tp1"
            if not (moe_dir / filename).is_file():
                gaps.append("tp1/moe.csv")
            continue
        present = csv_layers(tp_dir / filename)
        if case["tp"] != 1:
            stable = {
                layer for layer in layers
                if architecture["catalog"][category][layer].get("tp_stable")
            }
            present |= csv_layers(case["bundle"] / "tp1" / filename) & stable
        missing = sorted(layers - present)
        gaps.extend(f"{filename}:{layer}" for layer in missing)
    return gaps


def cluster_config(root, case):
    spec = HARDWARE[case["hardware"]]
    model_config = json.loads((root / "configs/model" / f"{case['model']}.json").read_text())
    return {
        "num_nodes": 1,
        "link_bw": spec["link_bw_gbps"],
        "link_latency": spec["link_latency_ns"],
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0},
            "instances": [{
                "model_name": case["model"],
                "hardware": case["hardware"],
                "npu_mem": {
                    "mem_size": spec["npu_mem_gb"],
                    "mem_bw": spec["npu_mem_bw_gbps"],
                    "mem_latency": 0,
                },
                "num_npus": case["tp"],
                "tp_size": case["tp"],
                "pd_type": None,
                "dtype": runtime_dtype(case["variant"]),
                "max_num_seqs": spec["max_num_seqs"],
                "max_num_batched_tokens": spec["max_num_batched_tokens"],
                "enable_prefix_caching": model_config.get("model_type") != "qwen3_next",
            }],
        }],
    }


def runtime_dtype(variant):
    if variant.startswith("fp16"):
        return "float16"
    return "bfloat16"


def stack_problems(root):
    problems = []
    for module in ("pandas", "pyinstrument", "rich", "numpy", "chakra", "google.protobuf"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            problems.append(f"missing Python module: {module}")
    binary = root / "astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        problems.append(f"missing analytical binary: {binary}")
    return problems


def write_summary(path, rows):
    fields = [
        "hardware", "model", "variant", "tp", "status", "profile_gaps", "returncode", "log", "output",
        "gpu_mem_used_fraction", "vram_headroom_gb", "kv_cache_util", "live_batch_size", "depth_req_q",
        "kvcache_hit_rate", "per_tok_comm_bytes", "kv_pressure_score_peak",
        "prefill_iteration_counts_per_second", "decode_itr_counts_per_second",
        "input_length_observed", "output_length_observed", "expert_inbalance", "pd_inbalance",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_SIM_METRICS_RE = re.compile(r"^SIM_METRICS instance=(\d+) (.+)$")
_SIM_METRICS_SUMMARY_RE = re.compile(r"^SIM_METRICS_SUMMARY instance=(\d+) (.+)$")


def read_sim_metrics(path, instance_id=0):
    """Return the final machine-readable simulator gauges for one instance."""
    metrics = {}
    if not path.is_file():
        return metrics
    for line in path.read_text().splitlines():
        match = _SIM_METRICS_RE.match(line)
        if match is None or int(match.group(1)) != instance_id:
            continue
        metrics = {
            key: "" if value == "N/A" else value
            for key, value in (token.split("=", 1) for token in match.group(2).split())
        }
    return metrics


def read_sim_metric_summary(path, instance_id=0):
    """Return the final machine-readable simulator summary for one instance."""
    metrics = {}
    if not path.is_file():
        return metrics
    for line in path.read_text().splitlines():
        match = _SIM_METRICS_SUMMARY_RE.match(line)
        if match is None or int(match.group(1)) != instance_id:
            continue
        metrics = dict(token.split("=", 1) for token in match.group(2).split())
    return metrics


def run_case(root, run_root, index, total, case, timeout):
    name = f"{index:03d}-{case['hardware']}-tp{case['tp']}"
    case_root = run_root / name
    case_root.mkdir(parents=True)
    config = case_root / "cluster.json"
    output = case_root / "requests.csv"
    log = case_root / "simulator.log"
    config.write_text(json.dumps(cluster_config(root, case), indent=2) + "\n")
    gaps = profile_gaps(root, case)
    command = [
        sys.executable, "-m", "serving",
        "--cluster-config", str(config.relative_to(root)),
        "--dataset", "workloads/example_trace.jsonl",
        "--num-reqs", "1",
        "--output", str(output),
        "--run-id", name,
        "--log-level", "WARNING",
    ]
    result = {**case, "profile_gaps": ";".join(gaps), "log": str(log.relative_to(root)),
              "output": str(output.relative_to(root)), "returncode": ""}
    print(f"[{index}/{total}] {case['hardware']} {case['model']} {case['variant']} tp{case['tp']}", flush=True)
    try:
        with log.open("w") as f:
            process = subprocess.Popen(
                command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, "FORCE_COLOR": "1"},
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            started = time.monotonic()
            while process.poll() is None:
                for _, _ in selector.select(timeout=1):
                    line = process.stdout.readline()
                    f.write(line)
                    print(line, end="", flush=True)
                if time.monotonic() - started > timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(command, timeout)
            for line in process.stdout:
                f.write(line)
                print(line, end="", flush=True)
            result["returncode"] = process.wait()
        result["status"] = "PASS_PARTIAL_PROFILE" if gaps else "PASS"
        if result["returncode"] or not output.is_file() or output.stat().st_size == 0:
            result["status"] = "FAIL"
        result.update(read_sim_metrics(log))
        result.update(read_sim_metric_summary(log))
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run all discovered cases")
    parser.add_argument("--list", action="store_true", help="list discovered cases")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per simulation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if yaml is None:
        print("STACK_NOT_READY: missing Python module: yaml", file=sys.stderr)
        return 2
    cases = list(profile_cases(root))
    if args.list or not args.run:
        for case in cases:
            gaps = profile_gaps(root, case)
            suffix = f" partial={','.join(gaps)}" if gaps else ""
            print(f"{case['hardware']} {case['model']} {case['variant']} tp{case['tp']}{suffix}")
        print(f"cases={len(cases)}")
        return 0
    problems = stack_problems(root)
    if problems:
        print("STACK_NOT_READY:\n" + "\n".join(f"- {item}" for item in problems), file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "outputs/smoke" / stamp
    rows = [run_case(root, run_root, index, len(cases), case, args.timeout)
            for index, case in enumerate(cases, 1)]
    write_summary(run_root / "summary.csv", rows)
    failures = [row for row in rows if row["status"] in {"FAIL", "TIMEOUT"}]
    print(f"summary={run_root / 'summary.csv'} cases={len(rows)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
