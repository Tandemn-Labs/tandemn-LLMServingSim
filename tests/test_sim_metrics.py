import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.smoke_test import read_sim_metric_summary, read_sim_metrics
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


if __name__ == "__main__":
    unittest.main()
