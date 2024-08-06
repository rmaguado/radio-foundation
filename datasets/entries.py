import os
import numpy as np
import json
from tqdm import tqdm
import random
import sys


def create_subsets(data, proportions):
    random.shuffle(data)
    
    num_items = len(data)
    start_index = 0
    subsets = {}
    
    for subset_name, proportion in proportions.items():
        subset_size = int(proportion * num_items)
        end_index = start_index + subset_size
        subsets[subset_name] = data[start_index:end_index]
        start_index = end_index
    
    if start_index < num_items:
        subsets[list(proportions.keys())[0]].extend(data[start_index:])
    
    return subsets

def main(dataset_name):
    base_dir = os.path.join(dataset_name, "data")
    target_path = os.path.join(dataset_name, "extra")
    os.makedirs(target_path, exist_ok=True)
    
    series_ids = [
        d for d in os.listdir(base_dir) \
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    print("Counting slices...")
    slice_counts = dict()
    for series_id in tqdm(series_ids):
        metadata_path = os.path.join(base_dir, series_id, "metadata.json")
        with open(metadata_path) as json_file:
            json_data = json.load(json_file)
        slice_counts[series_id] = json_data["shape"][0]

    splits = {
        "train": 0.8,
        "test": 0.2,
    }
    split_ids = create_subsets(series_ids, splits)

    dtype = np.dtype(
        [
            ("series_id", "U256"),
            ("slice_index", "uint16"),
        ]
    )
    for split in splits.keys():
        print(f"subset: {split}.")
        series_dirs = split_ids[split]
        split_slice_counts = [slice_counts[x] for x in split_ids[split]]
        length = sum(split_slice_counts)

        entries_array = np.empty(length, dtype=dtype)

        abs_slice_index = 0
        for series_id in tqdm(series_dirs):
            metadata_path = os.path.join(base_dir, series_id, "metadata.json")
            with open(metadata_path) as json_file:
                json_data = json.load(json_file)
            num_slices = json_data["shape"][0]
            for slice_index in range(num_slices):
                entries_array[abs_slice_index] = (
                    series_id,
                    slice_index
                )
                abs_slice_index += 1

        extra_full_path = os.path.join(target_path, f"entries-{split.upper()}.npy")
        np.save(extra_full_path, entries_array)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)
