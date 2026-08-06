def get_mps_command_queue() -> int:
    """PyTorch's MPS ``MTLCommandQueue*`` as a raw pointer, or 0 if unavailable.

    The pointer is borrowed: it stays valid for the lifetime of the PyTorch MPS
    runtime.
    """

__all__ = ["get_mps_command_queue"]
