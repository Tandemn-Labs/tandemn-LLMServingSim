import os
import re
import selectors
import subprocess
from time import monotonic
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self._stdout_buffer = ""
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def _read_line(self, p, deadline=None):
        while "\n" not in self._stdout_buffer:
            timeout = None if deadline is None else max(0, deadline - monotonic())
            if timeout is not None:
                ready = selectors.DefaultSelector()
                try:
                    ready.register(p.stdout, selectors.EVENT_READ)
                    if not ready.select(timeout):
                        return None
                finally:
                    ready.close()
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError(f"ASTRA-Sim exited unexpectedly with code {p.poll()}")
            self._stdout_buffer += chunk.decode()
        line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
        return line + "\n"

    def read_wait(self, p, timeout=None):
        """Read through the next ASTRA-Sim wait point, or return None on timeout."""
        out = [""]
        deadline = monotonic() + timeout if timeout is not None else None
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = self._read_line(p, deadline)
            if line is None:
                return None
            out.append(line)
        return out

    def stop(self, p, grace_seconds=5):
        """Stop ASTRA-Sim after an intentional partial simulation cutoff."""
        if p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    def check_end(self, p, timeout=5):
        deadline = monotonic() + timeout
        out = []
        while True:
            line = self._read_line(p, deadline)
            if line is None:
                raise TimeoutError("ASTRA-Sim did not finish within the completion grace period")
            out.append(line)
            if line == "All Request Has Been Exited\n":
                print(out[-2] if len(out) > 1 else "", end='')
                print(line, end='')
                return out
            if line == "ERROR: Some Requests Remain\n":
                raise RuntimeError("ASTRA-Sim reported unfinished requests")

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_waiting(self, outputs):
        """Return the NPU and simulator cycle waiting for a controller command."""
        for line in reversed(outputs):
            match = re.search(r"Waiting sys\[(\d+)\] cycle (\d+)", line)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None

    def parse_output(self, output):
        pattern = (
            r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
            r"exposed communication (\d+) cycles\."
            r"(?: iteration elapsed (\d+) cycles, active compute (\d+) cycles, "
            r"active communication (\d+) cycles"
            r"(?:, active GPU (\d+) cycles)?\.)?"
        )
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))
            iteration_elapsed = int(match.group(5)) if match.group(5) else None
            compute_active = int(match.group(6)) if match.group(6) else None
            comm_active = int(match.group(7)) if match.group(7) else None
            gpu_active = int(match.group(8)) if match.group(8) else None

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {
                'sys': sys,
                'id': id,
                'cycle': cycle,
                'iteration_elapsed': iteration_elapsed,
                'compute_active': compute_active,
                'comm_active': comm_active,
                'gpu_active': gpu_active,
            }
        return

    def parse_outputs(self, outputs):
        """Return every NPU timing report emitted before one Waiting line."""
        return [parsed for line in outputs if (parsed := self.parse_output(line))]
