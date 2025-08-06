import os
import time
import torch
import random
import torch.multiprocessing as mp
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from typing import Callable, Optional, Any, Iterator


class SequentialSampler:
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler:
    """Samples elements randomly. Without replacement."""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


def _worker_loop(
    dataset: Dataset, index_queue: mp.Queue, data_queue: mp.Queue, worker_id: int
):
    while True:
        try:
            sub_batch_indices = index_queue.get(timeout=5)

            if sub_batch_indices is None:
                break

            if not sub_batch_indices:
                data_queue.put([])
                continue

            samples = [dataset[i] for i in sub_batch_indices]
            data_queue.put(samples)

        except Exception as e:
            data_queue.put(e)
            break


class _CollaborativeLoaderIter:
    """The internal iterator that manages workers and yields batches."""

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.collate_fn = loader.collate_fn
        self.drop_last = loader.drop_last
        self.sampler = loader.sampler

        self.sampler_iter = iter(self.sampler)
        self._shutdown = False

        self.index_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.data_queue = mp.Queue()

        self.workers = []
        for i in range(self.num_workers):
            worker = mp.Process(
                target=_worker_loop,
                args=(self.dataset, self.index_queues[i], self.data_queue, i),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

    def __next__(self) -> Any:
        batch_indices = []
        for _ in range(self.batch_size):
            try:
                index = next(self.sampler_iter)
                batch_indices.append(index)
            except StopIteration:
                break

        if not batch_indices:
            self.shutdown()
            raise StopIteration

        if self.drop_last and len(batch_indices) < self.batch_size:
            self.shutdown()
            raise StopIteration

        sub_batches_indices = [[] for _ in range(self.num_workers)]
        for i, idx in enumerate(batch_indices):
            sub_batches_indices[i % self.num_workers].append(idx)

        active_workers = 0
        for i in range(self.num_workers):
            if sub_batches_indices[i]:
                self.index_queues[i].put(sub_batches_indices[i])
                active_workers += 1

        all_samples = []
        for _ in range(active_workers):
            result = self.data_queue.get()
            if isinstance(result, Exception):
                self.shutdown()
                raise RuntimeError(f"A worker process failed: {result}") from result
            all_samples.extend(result)

        return self.collate_fn(all_samples)

    def shutdown(self):
        """Cleanly terminates all worker processes."""
        if self._shutdown:
            return
        self._shutdown = True

        try:
            for q in self.index_queues:
                q.put(None)
            for w in self.workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()
        finally:
            for q in self.index_queues:
                q.close()
            self.data_queue.close()

    def __del__(self):
        self.shutdown()


class CollaborativeLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        sampler: Optional[Sampler] = None,
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

        # If a sampler is provided, it overrides the shuffle argument.
        if sampler is None:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
        else:
            if shuffle:
                print(
                    "Warning: Both a sampler and shuffle=True are specified. The sampler will be used."
                )
            self.sampler = sampler

    def __len__(self) -> int:
        """Returns the number of batches in the loader."""
        try:
            num_samples = len(self.sampler)  # type: ignore
            if self.drop_last:
                return num_samples // self.batch_size
            else:
                # Ceiling division
                return (num_samples + self.batch_size - 1) // self.batch_size
        except TypeError:
            raise TypeError(
                f"'{type(self.sampler).__name__}' instance doesn't have a __len__ method, "
                "so the length of this dataloader is not defined."
            )

    def __iter__(self) -> _CollaborativeLoaderIter:
        return _CollaborativeLoaderIter(self)


class SlowDataset(Dataset):
    def __init__(self, num_samples=100, delay=0.1):
        self.num_samples = num_samples
        self.delay = delay

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        time.sleep(self.delay)
        # Return index and the process ID of the worker
        return torch.tensor([float(idx), float(os.getpid())])


if __name__ == "__main__":

    BATCH_SIZE = 4
    NUM_WORKERS = 4
    DATASET_SIZE = 16

    slow_dataset = SlowDataset(num_samples=DATASET_SIZE, delay=1.0)

    infinite_loader = CollaborativeLoader(
        dataset=slow_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    t0 = time.time()
    for i, batch in enumerate(infinite_loader):  # type: ignore
        tf = time.time()
        print(f"Waited {tf - t0:.04f} seconds.")
        t0 = time.time()
        if i >= 5:
            break
    epoch_end_time = time.time()
