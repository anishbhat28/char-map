from __future__ import annotations
from collections import OrderedDict
from .types import Request, Transfer, AccessResult

class PEState:
    def __init__(self, pe_id: int, capacity: int):
        self.pe_id = pe_id
        self.capacity = capacity
        self.resident = OrderedDict()
        self.prefetched_blocks = set()

    def contains(self, block_id: int) -> bool:
        return block_id in self.resident

    def touch(self, block_id: int, cycle: int):
        if block_id in self.resident:
            self.resident.move_to_end(block_id)
            self.resident[block_id] = cycle

    def insert(self, block_id: int, cycle: int, prefetched: bool = False):
        if block_id in self.resident:
            self.resident.move_to_end(block_id)
            self.resident[block_id] = cycle
            if prefetched:
                self.prefetched_blocks.add(block_id)
            return None
        evicted = None
        if self.capacity <= 0:
            return block_id
        if len(self.resident) >= self.capacity:
            evicted, _ = self.resident.popitem(last=False)
            self.prefetched_blocks.discard(evicted)
        self.resident[block_id] = cycle
        if prefetched:
            self.prefetched_blocks.add(block_id)
        return evicted

    def was_prefetched(self, block_id: int) -> bool:
        return block_id in self.prefetched_blocks

    def consume_prefetch_mark(self, block_id: int):
        self.prefetched_blocks.discard(block_id)

class RingHardware:
    def __init__(self, num_pes: int, hop_latency: int, compute_latency: int, local_capacity: int, block_size_bytes: int):
        self.num_pes = num_pes
        self.hop_latency = hop_latency
        self.compute_latency = compute_latency
        self.local_capacity = local_capacity
        self.block_size_bytes = block_size_bytes
        self.pes = [PEState(i, local_capacity) for i in range(num_pes)]
        self.inflight = []
        self.total_bytes = 0
        self.total_hops = 0
        self.total_prefetch_bytes = 0
        self.evictions = 0

    def ring_distance(self, a: int, b: int) -> int:
        raw = abs(a - b)
        return min(raw, self.num_pes - raw)

    def initialize_home_blocks(self, num_blocks: int):
        for block in range(num_blocks):
            pe = block % self.num_pes
            ev = self.pes[pe].insert(block, 0, False)
            if ev is not None and ev != block:
                self.evictions += 1

    def advance(self, cycle: int):
        completed = [tr for tr in self.inflight if tr.finish <= cycle]
        self.inflight = [tr for tr in self.inflight if tr.finish > cycle]
        for tr in completed:
            ev = self.pes[tr.dst].insert(tr.block_id, tr.finish, tr.prefetched)
            if ev is not None and ev != tr.block_id:
                self.evictions += 1

    def _existing_inflight(self, block_id: int, dst: int):
        xs = [tr for tr in self.inflight if tr.block_id == block_id and tr.dst == dst]
        return min(xs, key=lambda x: x.finish) if xs else None

    def schedule_transfer(self, block_id: int, src: int, dst: int, start: int, prefetched: bool):
        if self.pes[dst].contains(block_id):
            return None
        existing = self._existing_inflight(block_id, dst)
        if existing is not None:
            return existing
        hops = self.ring_distance(src, dst)
        finish = start + hops * self.hop_latency
        tr = Transfer(block_id, src, dst, start, finish, self.block_size_bytes, prefetched)
        self.inflight.append(tr)
        self.total_bytes += self.block_size_bytes
        self.total_hops += hops
        if prefetched:
            self.total_prefetch_bytes += self.block_size_bytes
        return tr

    def schedule_prefetch(self, block_id: int, src: int, dst: int, cycle: int):
        return self.schedule_transfer(block_id, src, dst, cycle, True)

    def serve(self, request: Request, cycle: int) -> AccessResult:
        self.advance(cycle)
        pe = self.pes[request.consumer_pe]
        if pe.contains(request.block_id):
            ph = pe.was_prefetched(request.block_id)
            pe.touch(request.block_id, cycle)
            if ph:
                pe.consume_prefetch_mark(request.block_id)
            return AccessResult(True, 0, 0, 0, ph)

        existing = self._existing_inflight(request.block_id, request.consumer_pe)
        if existing is not None:
            stall = max(0, existing.finish - cycle)
            self.advance(existing.finish)
            self.pes[request.consumer_pe].touch(request.block_id, existing.finish)
            if existing.prefetched:
                self.pes[request.consumer_pe].consume_prefetch_mark(request.block_id)
            return AccessResult(False, stall, self.ring_distance(existing.src, existing.dst), 0, existing.prefetched)

        src = request.producer_pe
        hops = self.ring_distance(src, request.consumer_pe)
        tr = self.schedule_transfer(request.block_id, src, request.consumer_pe, cycle, False)
        finish = tr.finish if tr else cycle
        stall = max(0, finish - cycle)
        self.advance(finish)
        self.pes[request.consumer_pe].touch(request.block_id, finish)
        return AccessResult(False, stall, hops, self.block_size_bytes if tr else 0, False)
