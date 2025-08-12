import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, Sampler
from queue import Empty
import threading
import uuid
from typing import Optional
import time


def _sample_worker_loop(dataset, task_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        batch_id, idx, result_queue = task
        try:
            sample = dataset[idx]
            result_queue.put((batch_id, idx, sample))
        except Exception as e:
            result_queue.put((batch_id, idx, e))


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
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.timeout = timeout

        self.sampler = sampler or list(range(len(dataset)))
        self.task_queue = mp.Queue()
        self.workers = []
        self._init_workers()
        self._shutdown_called = False

    def _init_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(
                target=_sample_worker_loop,
                args=(self.dataset, self.task_queue),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def _batch_collector(self, batch_id, indices, result_queue):
        samples = {}
        pending = set(indices)
        try:
            while pending:
                bid, idx, sample = result_queue.get(timeout=self.timeout)
                if bid != batch_id:
                    continue
                if isinstance(sample, Exception):
                    raise sample
                samples[idx] = sample
                pending.remove(idx)
        except Empty:
            raise RuntimeError("Sample worker timeout")

        ordered = [samples[i] for i in indices]
        batch = self.collate_fn(ordered)
        self.batch_queue.put(batch)

    def __iter__(self):
        self._iterator = iter(self.sampler)
        self._stop_event = threading.Event()
        self.batch_queue = mp.Queue(maxsize=self.prefetch_batches)
        self._manager = mp.Manager()
        self._start_prefetch_threads()
        return self

    def _start_prefetch_threads(self):
        self.prefetch_threads = []

        def spawn_batches():
            try:
                while not self._stop_event.is_set():
                    indices = [next(self._iterator) for _ in range(self.batch_size)]
                    batch_id = uuid.uuid4().hex
                    result_queue = self._manager.Queue()

                    for idx in indices:
                        self.task_queue.put((batch_id, idx, result_queue))

                    t = threading.Thread(
                        target=self._batch_collector,
                        args=(batch_id, indices, result_queue),
                        daemon=True,
                    )
                    t.start()
                    self.prefetch_threads.append(t)

                    while (
                        len([t for t in self.prefetch_threads if t.is_alive()])
                        >= self.prefetch_batches
                    ):
                        time.sleep(0.01)

            except StopIteration:
                pass
            finally:
                self._stop_event.set()

        self._prefetch_controller = threading.Thread(target=spawn_batches, daemon=True)
        self._prefetch_controller.start()

    def __next__(self):
        if self._shutdown_called:
            raise StopIteration

        try:
            batch = self.batch_queue.get(timeout=self.timeout)
            return batch
        except Empty:
            if self._prefetch_controller.is_alive() or any(
                t.is_alive() for t in self.prefetch_threads
            ):
                raise RuntimeError("Batch queue timeout, but workers are still active.")
            else:
                self._shutdown()
                raise StopIteration

    def _shutdown(self):
        if self._shutdown_called:
            return

        self._shutdown_called = True
        self._stop_event.set()

        if self._prefetch_controller.is_alive():
            self._prefetch_controller.join()

        for t in self.prefetch_threads:
            if t.is_alive():
                t.join(timeout=self.timeout)

        for _ in self.workers:
            self.task_queue.put(None)

        for w in self.workers:
            if w.is_alive():
                w.join(timeout=self.timeout)

        self.task_queue.close()
        self.batch_queue.close()
        self._manager.shutdown()

    def __del__(self):
        self._shutdown()


class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = list(range(size))

    def __getitem__(self, index):
        time.sleep(0.01)
        return self.data[index] * 2

    def __len__(self):
        return len(self.data)


def simple_collate_fn(batch):
    return torch.tensor(batch)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    dataset = DummyDataset(100)
    dataloader = ParallelSampleDataLoader(
        dataset,
        batch_size=2,
        collate_fn=simple_collate_fn,
        num_workers=8,
        prefetch_batches=4,
    )

    t0 = time.time()
    for idx, batch in enumerate(dataloader):
        tf = time.time() - t0
        print(f"{idx} | {tf:.4f} sec | {batch}")
        t0 = time.time()

    dataloader._shutdown()
