import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from scripts.smoke_test import read_sim_metric_summary, read_sim_metrics
sys.modules.setdefault("pyinstrument", SimpleNamespace(Profiler=object))
from serving.__main__ import (
    _instance_metrics,
    _open_metrics_file,
    _prefix_cache_hit_rate,
    _print_instance_metrics,
    _write_metrics_sample,
)
from serving.core.controller import Controller
from serving.core.memory_model import MemoryModel
from serving.core.scheduler import Scheduler
from serving.core.trace_generator import _trace_collective_wire_bytes


class SimMetricsTest(unittest.TestCase):
    def test_total_kv_for_tokens_is_block_rounded(self):
        memory = MemoryModel.__new__(MemoryModel)
        memory.block_size = 16
        memory.is_qwen3_next = False
        memory.get_kv = lambda tokens: tokens * 10

        self.assertEqual(memory.get_total_kv_for_tokens(0), 0)
        self.assertEqual(memory.get_total_kv_for_tokens(17), 32 * 10)

    def test_qwen_next_adds_recurrent_state_once(self):
        memory = MemoryModel.__new__(MemoryModel)
        memory.block_size = 16
        memory.is_qwen3_next = True
        memory.get_kv = lambda tokens: tokens * 10
        memory._qwen3_next_state_bytes = lambda: 123

        self.assertEqual(memory.get_total_kv_for_tokens(17), 32 * 10 + 123)

    def test_kv_pressure_deduplicates_inflight_request(self):
        request_a = SimpleNamespace(id=1, output=20)
        request_b = SimpleNamespace(id=2, output=30)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.request = [request_a]
        scheduler.inflight = [SimpleNamespace(requests=[request_a, request_b])]
        scheduler.memory = SimpleNamespace(
            npu_mem=1_100,
            weight=100,
            get_total_kv_for_tokens=lambda tokens: tokens * 10,
        )

        self.assertEqual(scheduler.get_kv_pressure_score(), 0.5)

    def test_trace_collectives_use_tp_and_ep_group_sizes(self):
        rows = [
            ["layer", "1", "LOCAL", "0", "LOCAL", "0", "LOCAL", "0",
             "ALLREDUCE", "100", "NONE"],
            ["EXPERT", "0", "ALLGATHER", "80"],
            ["EXPERT", "END", "REDUCESCATTER", "80"],
        ]

        self.assertEqual(_trace_collective_wire_bytes(rows, tp_size=4, ep_total=2), 230)

    def test_controller_parses_all_npu_timing_reports(self):
        controller = Controller(2)
        reports = controller.parse_outputs([
            "sys[1] iteration 2 finished, 100 cycles, exposed communication 30 cycles. "
            "iteration elapsed 40 cycles, active compute 25 cycles, "
            "active communication 12 cycles, active GPU 31 cycles.\n",
            "sys[0] iteration 2 finished, 100 cycles, exposed communication 20 cycles. "
            "iteration elapsed 40 cycles, active compute 30 cycles, "
            "active communication 8 cycles, active GPU 35 cycles.\n",
            "Waiting\n",
        ])

        self.assertEqual([report["sys"] for report in reports], [1, 0])
        self.assertEqual(sum(report["comm_active"] for report in reports), 20)
        self.assertEqual(sum(report["gpu_active"] for report in reports), 66)

    def test_smoke_parser_merges_live_and_summary_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "simulator.log"
            log.write_text(
                "SIM_METRICS instance=0 comm_overhead_pct=0.250000 "
                "pipeline_bubble_fraction=0.125000\n"
                "SIM_METRICS_SUMMARY instance=0 kv_pressure_score_peak=0.750000\n"
            )

            metrics = read_sim_metrics(log)
            metrics.update(read_sim_metric_summary(log))

        self.assertEqual(metrics["comm_overhead_pct"], "0.250000")
        self.assertEqual(metrics["pipeline_bubble_fraction"], "0.125000")
        self.assertEqual(metrics["kv_pressure_score_peak"], "0.750000")

    def test_metrics_jsonl_preserves_sample_metadata_and_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                _write_metrics_sample(
                    handle,
                    10_000_000_000,
                    "periodic",
                    2,
                    {"kv_cache_util": 0.5, "expert_inbalance": None},
                )
                _write_metrics_sample(
                    handle,
                    12_000_000_000,
                    "final",
                    2,
                    {"kv_cache_util": 0.75},
                )
            periodic, final = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(periodic["sim_time_ns"], 10_000_000_000)
        self.assertEqual(periodic["sample_kind"], "periodic")
        self.assertEqual(periodic["instance_id"], 2)
        self.assertIsNone(periodic["expert_inbalance"])
        self.assertEqual(final["sample_kind"], "final")
        self.assertEqual(final["kv_cache_util"], 0.75)
        self.assertNotIn("kv_pressure_score_peak", final)

    def test_metrics_output_requires_existing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "metrics.jsonl"
            with self.assertRaisesRegex(ValueError, "parent does not exist"):
                _open_metrics_file(path)
            self.assertFalse(path.parent.exists())

    def test_metrics_output_is_caller_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            handle = _open_metrics_file(path)
            _write_metrics_sample(handle, 0, "final", 0, {})
            handle.close()

            self.assertTrue(path.is_file())

    def test_prefix_cache_hit_rate_uses_interval_deltas(self):
        self.assertEqual(_prefix_cache_hit_rate(300, 120, 100, 20), 0.5)
        self.assertIsNone(_prefix_cache_hit_rate(300, 120, 300, 120))
        self.assertIsNone(_prefix_cache_hit_rate(300, 20, 100, 30))

    def test_unimplemented_imbalance_is_null_in_metrics_not_stdout(self):
        scheduler = SimpleNamespace(
            memory=SimpleNamespace(npu_mem=100.0, weight=40.0, npu_used=50.0),
            enable_prefix_caching=False,
            inflight=[],
            request=[],
            pp_size=1,
            get_kv_pressure_score=lambda: 0.0,
        )
        with redirect_stdout(io.StringIO()) as output:
            metrics = _print_instance_metrics(0, scheduler, 0, kvcache_hit_rate=0.5)

        self.assertIsNone(metrics["expert_inbalance"])
        self.assertIsNone(metrics["pd_inbalance"])
        self.assertEqual(metrics["kvcache_hit_rate"], 0.5)
        self.assertIn("expert_inbalance=0.000000 pd_inbalance=0.000000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
