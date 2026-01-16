import torch
import torch.multiprocessing as mp

from logs import get_logger

_LOGGER = get_logger(__name__)


class SharedParamsStore:
    def __init__(self, model: torch.nn.Module):
        vec_cpu = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        self.shared = torch.empty_like(vec_cpu).share_memory_()
        self.shared.copy_(vec_cpu)

        self.tmp_cpu = torch.empty_like(vec_cpu)  # reusable staging buffer

        self.version = mp.Value("q", 0)

    @torch.no_grad()
    def publish(self, model: torch.nn.Module):
        _LOGGER.debug("Publishing parameters to shared memory")
        vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        self.tmp_cpu.copy_(vec, non_blocking=False)

        self.version.value += 1  # odd: writing
        self.shared.copy_(self.tmp_cpu)
        self.version.value += 1  # even: committed

    @torch.no_grad()
    def read_into(self, out_vec: torch.Tensor):
        _LOGGER.debug("Reading parameters from shared memory")
        out_vec.copy_(self.shared)
