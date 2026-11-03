"""QwenQPU: fixed-gate ideal state-vector module with an explicit host runtime."""
from .core import LocalGateStatevector, encode
from .runtime import QwenQPU, execute_packet

__version__ = "0.1.0.dev0"
__all__ = ["LocalGateStatevector", "encode", "QwenQPU", "execute_packet"]
