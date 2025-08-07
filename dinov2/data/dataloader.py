import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, Sampler
from queue import Empty
import threading
import uuid


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
        dataset: Dataset,
        batch_size: int,
        collate_fn,
        num_workers: int = 4,
        sampler: Sampler = None,
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
                    continue  # skip stray messages
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
        self._start_prefetch_threads()
        return self

    def _start_prefetch_threads(self):
        self.prefetch_threads = []

        def spawn_batches():
            while not self._stop_event.is_set():
                try:
                    indices = [next(self._iterator) for _ in range(self.batch_size)]
                except StopIteration:
                    break

                batch_id = uuid.uuid4().hex
                if not hasattr(self, "_manager"):
                    self._manager = mp.Manager()
                    result_queue = self._manager.Queue()
                else:
                    result_queue = mp.Queue()

                for idx in indices:
                    self.task_queue.put((batch_id, idx, result_queue))

                t = threading.Thread(
                    target=self._batch_collector,
                    args=(batch_id, indices, result_queue),
                    daemon=True,
                )
                t.start()
                self.prefetch_threads.append(t)

                # Limit number of concurrent batches
                while len(self.prefetch_threads) >= self.prefetch_batches:
                    self.prefetch_threads = [
                        t for t in self.prefetch_threads if t.is_alive()
                    ]
                    if len(self.prefetch_threads) >= self.prefetch_batches:
                        threading.Event().wait(0.01)

        self._prefetch_controller = threading.Thread(target=spawn_batches, daemon=True)
        self._prefetch_controller.start()

    def __next__(self):
        if self.batch_queue.empty() and not self._prefetch_controller.is_alive():
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=self.timeout)
        except Empty:
            raise StopIteration

    def __del__(self):
        self._stop_event.set()
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            if w.is_alive():
                w.terminate()


import time
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = list(range(size))

    def __getitem__(self, index):
        time.sleep(0.5)
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
    for batch in dataloader:
        tf = time.time() - t0
        print(f"{tf:.4f} sec | {batch}")
        t0 = time.time()
