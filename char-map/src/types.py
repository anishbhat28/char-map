from dataclasses import dataclass

@dataclass(frozen=True)
class Request:
    timestep: int
    consumer_pe: int
    block_id: int
    producer_pe: int
    required_at: int

@dataclass(frozen=True)
class Prediction:
    block_id: int
    target_pe: int
    needed_at: int
    confidence: float = 1.0

@dataclass
class Transfer:
    block_id: int
    src: int
    dst: int
    start: int
    finish: int
    bytes_moved: int
    prefetched: bool = False

@dataclass
class AccessResult:
    hit: bool
    stall_cycles: int
    transfer_hops: int
    bytes_transferred: int
    prefetched_hit: bool
