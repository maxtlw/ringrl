from multiprocessing import shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock as LockType

import numpy as np

from logs import get_logger

_LOGGER = get_logger(__name__)


class SharedReplayRing:
    """
    Shared-memory ring storing transitions.
    obs/next_obs are uint8 Atari stacks (4,84,84) by default.
    """

    # Set on `attach(...)` (and shared between processes)
    write_pos: Synchronized
    lock: LockType

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int] = (4, 84, 84)):
        self.capacity = capacity
        self.obs_shape = obs_shape

        obs_nbytes = (
            np.prod((self.capacity, *self.obs_shape)) * np.dtype(np.uint8).itemsize
        )
        act_nbytes = self.capacity * np.dtype(np.int64).itemsize
        rew_nbytes = self.capacity * np.dtype(np.float32).itemsize
        done_nbytes = (
            self.capacity * np.dtype(np.uint8).itemsize
        )  # store as uint8 for simplicity
        seq_nbytes = (
            capacity * np.dtype(np.uint32).itemsize
        )  # sequence numbers for commit protocol

        _LOGGER.debug(
            f"Creating shared memory for transitions: {obs_nbytes / 1024 / 1024:.2f} MB"
        )

        # Create shared memory for transitions
        self.shm_obs = shared_memory.SharedMemory(create=True, size=int(obs_nbytes))
        self.shm_next_obs = shared_memory.SharedMemory(
            create=True, size=int(obs_nbytes)
        )
        self.shm_act = shared_memory.SharedMemory(create=True, size=int(act_nbytes))
        self.shm_rew = shared_memory.SharedMemory(create=True, size=int(rew_nbytes))
        self.shm_terminated = shared_memory.SharedMemory(
            create=True, size=int(done_nbytes)
        )
        self.shm_truncated = shared_memory.SharedMemory(
            create=True, size=int(done_nbytes)
        )
        self.shm_seq = shared_memory.SharedMemory(create=True, size=int(seq_nbytes))

        self._build_views()

    def _build_views(self):
        self.obs = np.ndarray(
            (self.capacity, *self.obs_shape), dtype=np.uint8, buffer=self.shm_obs.buf
        )
        self.next_obs = np.ndarray(
            (self.capacity, *self.obs_shape),
            dtype=np.uint8,
            buffer=self.shm_next_obs.buf,
        )
        self.act = np.ndarray((self.capacity,), dtype=np.int64, buffer=self.shm_act.buf)
        self.rew = np.ndarray(
            (self.capacity,), dtype=np.float32, buffer=self.shm_rew.buf
        )
        self.terminated = np.ndarray(
            (self.capacity,), dtype=np.uint8, buffer=self.shm_terminated.buf
        )
        self.truncated = np.ndarray(
            (self.capacity,), dtype=np.uint8, buffer=self.shm_truncated.buf
        )
        self.seq = np.ndarray(
            (self.capacity,), dtype=np.uint32, buffer=self.shm_seq.buf
        )

    def export_handles(self) -> dict:
        """Pass this dict to child processes."""
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "shm_obs": self.shm_obs.name,
            "shm_next_obs": self.shm_next_obs.name,
            "shm_act": self.shm_act.name,
            "shm_rew": self.shm_rew.name,
            "shm_terminated": self.shm_terminated.name,
            "shm_truncated": self.shm_truncated.name,
            "shm_seq": self.shm_seq.name,
        }

    @staticmethod
    def attach(handles: dict, write_pos: Synchronized, lock: LockType):
        obj = object.__new__(SharedReplayRing)
        obj.capacity = int(handles["capacity"])
        obj.obs_shape = tuple(handles["obs_shape"])

        obj.shm_obs = shared_memory.SharedMemory(name=handles["shm_obs"])
        obj.shm_next_obs = shared_memory.SharedMemory(name=handles["shm_next_obs"])
        obj.shm_act = shared_memory.SharedMemory(name=handles["shm_act"])
        obj.shm_rew = shared_memory.SharedMemory(name=handles["shm_rew"])
        obj.shm_terminated = shared_memory.SharedMemory(name=handles["shm_terminated"])
        obj.shm_truncated = shared_memory.SharedMemory(name=handles["shm_truncated"])
        obj.shm_seq = shared_memory.SharedMemory(name=handles["shm_seq"])

        obj._build_views()
        obj.write_pos = write_pos
        obj.lock = lock
        return obj

    def close_and_unlink(self):
        for shm in [
            self.shm_obs,
            self.shm_next_obs,
            self.shm_act,
            self.shm_rew,
            self.shm_terminated,
            self.shm_truncated,
            self.shm_seq,
        ]:
            shm.close()
            shm.unlink()

    def close_only(self):
        for shm in [
            self.shm_obs,
            self.shm_next_obs,
            self.shm_act,
            self.shm_rew,
            self.shm_terminated,
            self.shm_truncated,
            self.shm_seq,
        ]:
            shm.close()

    def write_slot(
        self,
        obs_u8: np.ndarray,
        act: int,
        rew: float,
        next_obs_u8: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        """
        Commit protocol with seq numbers.
        """
        obs_u8 = np.ascontiguousarray(obs_u8, dtype=np.uint8)
        next_obs_u8 = np.ascontiguousarray(next_obs_u8, dtype=np.uint8)

        with self.lock:
            i = int(self.write_pos.value % self.capacity)
            self.write_pos.value += 1

        # Start write: make seq odd
        s = int(self.seq[i])
        if (s & 1) != 0:
            # Shouldn't happen if protocol is respected; but avoid deadlock
            s = s + 1
        self.seq[i] = s + 1  # odd = writing

        # Write payload
        self.obs[i] = obs_u8
        self.next_obs[i] = next_obs_u8
        self.act[i] = int(act)
        self.rew[i] = float(rew)
        self.terminated[i] = 1 if terminated else 0
        self.truncated[i] = 1 if truncated else 0

        self.seq[i] = s + 2  # even = committed

    def read_batch_safe(self, idxs: np.ndarray, max_spins: int = 20):
        """
        Reads a batch of indices. Retries individual slots if writer overlaps.
        Returns copies (safe) for obs/next_obs; scalars are copied too.
        """
        B = int(idxs.shape[0])
        obs = np.empty((B, *self.obs_shape), dtype=np.uint8)
        next_obs = np.empty((B, *self.obs_shape), dtype=np.uint8)
        act = np.empty((B,), dtype=np.int64)
        rew = np.empty((B,), dtype=np.float32)
        terminated = np.empty((B,), dtype=np.uint8)
        truncated = np.empty((B,), dtype=np.uint8)

        for j, i in enumerate(idxs.tolist()):
            for _ in range(max_spins):
                s1 = int(self.seq[i])
                if (s1 & 1) != 0:
                    continue  # writer in progress
                # read
                obs[j] = self.obs[i]
                next_obs[j] = self.next_obs[i]
                act[j] = self.act[i]
                rew[j] = self.rew[i]
                terminated[j] = self.terminated[i]
                truncated[j] = self.truncated[i]
                s2 = int(self.seq[i])
                if s1 == s2 and (s2 & 1) == 0:
                    break
            else:
                # As a fallback, just take what we got (rare). Or resample.
                pass

        return (
            obs,
            act,
            rew,
            next_obs,
            terminated.astype(np.bool_),
            truncated.astype(np.bool_),
        )
