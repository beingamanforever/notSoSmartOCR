"""Small runtime measurements shared by benchmark entrypoints."""

from __future__ import annotations


class CudaMonitor:
    """Measure PyTorch CUDA peak allocation when that runtime is available."""

    def __init__(self) -> None:
        try:
            import torch
        except (ImportError, OSError):
            self._torch = None
        else:
            self._torch = torch if torch.cuda.is_available() else None

    @property
    def available(self) -> bool:
        return self._torch is not None

    def begin(self) -> None:
        if self._torch is None:
            return
        self._torch.cuda.synchronize()
        self._torch.cuda.reset_peak_memory_stats()

    def finish(self) -> int | None:
        if self._torch is None:
            return None
        self._torch.cuda.synchronize()
        return int(self._torch.cuda.max_memory_allocated())
