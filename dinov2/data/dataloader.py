import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, Sampler
from queue import Empty
import threading
from typing import Optional, Dict, List, Tuple, Any
import itertools
import time
import collections.abc


def _recursively_share_memory(obj: Any) -> Any:
    """
    If obj is a torch.Tensor, call .share_memory_().
    If obj is a tuple/list/dict, walk recursively and share tensors in-place.
    Return the original object (modified).
    """
    if isinstance(obj, torch.Tensor):
        try:
            if not obj.is_shared():
                obj.share_memory_()
        except Exception:
            pass
        return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _recursively_share_memory(v)
        return obj
    elif isinstance(obj, (list, tuple)):
        items = [_recursively_share_memory(x) for x in obj]
        return type(obj)(items)
    else:
        return obj


def _recursively_pin_memory(obj: Any) -> Any:
    """
    If obj is a torch.Tensor, call .pin_memory().
    If obj is a tuple/list/dict, walk recursively and return a new structure
    with pinned tensors where possible.
    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.pin_memory()
        except Exception:
            return obj
    elif isinstance(obj, dict):
        return {k: _recursively_pin_memory(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursively_pin_memory(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_recursively_pin_memory(x) for x in obj)
    else:
        return obj


def _sample_worker_loop(
    dataset, index_queue, sample_queue, shared_memory: bool, pin_memory: bool
):
    """Worker process: read an index, fetch dataset[idx], push (idx, sample)."""
    while True:
        idx = index_queue.get()
        if idx is None:
            break
        try:
            sample = dataset[idx]
            if shared_memory:
                sample = _recursively_share_memory(sample)
            if pin_memory:
                sample = _recursively_pin_memory(sample)
            sample_queue.put((idx, sample))
        except Exception as e:
            sample_queue.put((idx, e))


class ParallelSampleDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        collate_fn,
        num_workers: int = 4,
        sampler: Optional[Sampler] = None,
        prefetch_batches: int = 2,
        timeout: float = 10.0,
        pin_memory: bool = False,
        shared_memory: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = max(1, prefetch_batches)
        self.timeout = float(timeout)
        self.pin_memory = bool(pin_memory)
        self.shared_memory = bool(shared_memory)

        self.sampler = sampler or list(range(len(dataset)))

        qsize = max(4, self.num_workers * 4)
        self.index_queue = mp.Queue(maxsize=qsize)
        self.sample_queue = mp.Queue(maxsize=qsize)
        self.workers: List[mp.Process] = []
        self._shutdown_called = False
        self._init_workers()

    def _init_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(
                target=_sample_worker_loop,
                args=(
                    self.dataset,
                    self.index_queue,
                    self.sample_queue,
                    self.shared_memory,
                    self.pin_memory,
                ),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def __iter__(self):
        if hasattr(self.sampler, "set_epoch"):
            try:
                self.sampler.set_epoch(0)
            except Exception:
                pass

        self._iterator = iter(self.sampler)
        self._stop_event = threading.Event()
        self.batch_queue = mp.Queue(maxsize=self.prefetch_batches)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )

        self._sampler_exhausted = False

        self._prefetch_thread.start()
        return self

    def _prefetch_loop(self):
        """
        Prefetch loop that:
         - creates batches of indices (up to prefetch_batches in flight),
         - submits indices to workers,
         - consumes sample results and assembles batches.
        """
        in_flight: Dict[int, Dict] = {}
        idx_to_pos: Dict[int, Tuple[int, int]] = {}
        batch_id_counter = itertools.count()
        sampler_exhausted = False

        try:
            while not self._stop_event.is_set():
                while len(in_flight) < self.prefetch_batches and not sampler_exhausted:
                    indices = []
                    try:
                        for _ in range(self.batch_size):
                            indices.append(next(self._iterator))
                    except StopIteration:
                        sampler_exhausted = True
                    if len(indices) == 0:
                        break

                    bid = next(batch_id_counter)
                    results = [None] * len(indices)
                    in_flight[bid] = {
                        "indices": indices,
                        "results": results,
                        "remaining": len(indices),
                    }

                    for pos, idx in enumerate(indices):
                        idx_to_pos[idx] = (bid, pos)
                        self.index_queue.put(idx)

                if not in_flight and sampler_exhausted:
                    for _ in range(self.num_workers):
                        self.index_queue.put(None)
                    break

                try:
                    s_idx, sample = self.sample_queue.get(timeout=self.timeout)
                except Empty:
                    if in_flight:
                        raise RuntimeError("Timeout waiting for a sample from workers.")
                    else:
                        break

                mapping = idx_to_pos.pop(s_idx, None)
                if mapping is None:
                    continue

                bid, pos = mapping
                entry = in_flight.get(bid)
                if entry is None:
                    continue

                if isinstance(sample, Exception):
                    raise sample

                entry["results"][pos] = sample
                entry["remaining"] -= 1

                if entry["remaining"] == 0:
                    ordered = entry["results"]
                    batch = self.collate_fn(ordered)

                    self.batch_queue.put(batch)
                    del in_flight[bid]

            while in_flight:
                try:
                    s_idx, sample = self.sample_queue.get(timeout=self.timeout)
                except Empty:
                    raise RuntimeError("Timeout while draining remaining samples.")
                mapping = idx_to_pos.pop(s_idx, None)
                if mapping is None:
                    continue
                bid, pos = mapping
                entry = in_flight.get(bid)
                if entry is None:
                    continue
                if isinstance(sample, Exception):
                    raise sample
                entry["results"][pos] = sample
                entry["remaining"] -= 1
                if entry["remaining"] == 0:
                    ordered = entry["results"]
                    batch = self.collate_fn(ordered)
                    if self.pin_memory:
                        batch = _recursively_pin_memory(batch)
                    self.batch_queue.put(batch)
                    del in_flight[bid]

        except Exception:
            self._stop_event.set()
            raise
        finally:
            self._stop_event.set()

    def __next__(self):
        if self._shutdown_called:
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=self.timeout)
        except Empty:
            self._shutdown()
            raise StopIteration

    def _shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._stop_event.set()

        if hasattr(self, "_prefetch_thread") and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=self.timeout)

        for _ in self.workers:
            self.index_queue.put(None)
        for w in self.workers:
            w.join(timeout=self.timeout)

        try:
            self.index_queue.close()
            self.sample_queue.close()
            self.batch_queue.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self._shutdown()
        except Exception:
            pass


class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = list(range(size))

    def __getitem__(self, index):
        time.sleep(0.05)
        return torch.tensor(self.data[index])

    def __len__(self):
        return len(self.data)


def simple_collate_fn(batch):
    return torch.stack(batch, dim=0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    dataset = DummyDataset(100)
    loader = ParallelSampleDataLoader(
        dataset,
        batch_size=2,
        collate_fn=simple_collate_fn,
        num_workers=4,
        prefetch_batches=4,
        timeout=30.0,
        pin_memory=False,
        shared_memory=True,
    )

    t_total = 0

    t0 = time.time()
    for idx, batch in enumerate(loader):
        tf = time.time() - t0
        t_total += tf
        print(f"{idx} | {tf:.4f} sec | {batch.shape}")
        t0 = time.time()

    print(t_total / 50)

    loader._shutdown()
