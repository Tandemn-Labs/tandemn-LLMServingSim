import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

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
