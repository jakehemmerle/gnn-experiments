"""Distributed training utilities for multi-GPU support."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int]:
    """Initialize distributed training if running under torchrun.

    Returns:
        Tuple of (rank, world_size)
        - rank: Process rank (0 if not distributed)
        - world_size: Total number of processes (1 if not distributed)
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def maybe_wrap_ddp(
    model: torch.nn.Module,
    rank: int,
    world_size: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Wrap model in DDP if distributed training is active.

    Args:
        model: The model to potentially wrap
        rank: Process rank
        world_size: Total number of processes

    Returns:
        Tuple of (wrapped_model, raw_model) - both are the same if not distributed
    """
    if world_size > 1:
        wrapped = DDP(model, device_ids=[rank])
        return wrapped, model
    return model, model


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
