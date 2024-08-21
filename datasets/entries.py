import os
import numpy as np
import json
from tqdm import tqdm
import random
import sys


def create_subsets(data, proportions):
    """
    Split the data into subsets based on the given proportions.
    """
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

def load_slice_counts(base_dir, series_ids):
    """
    Load the slice counts for each series from the metadata files.
    """
    slice_counts = {}
    for series_id in tqdm(series_ids, desc="Counting slices"):
        metadata_path = os.path.join(base_dir, series_id, "metadata.json")
        with open(metadata_path) as json_file:
            json_data = json.load(json_file)
        slice_counts[series_id] = json_data["shape"][0]
    
    return slice_counts

def prepare_entries_array(series_dirs, slice_counts, base_dir):
    """
    Prepare an array of entries for a given split.
    """
    total_slices = sum(slice_counts[series_id] for series_id in series_dirs)
    dtype = np.dtype([("series_id", "U256"), ("slice_index", "uint16")])
    entries_array = np.empty(total_slices, dtype=dtype)

    abs_slice_index = 0
    for series_id in tqdm(series_dirs, desc="Processing series"):
        metadata_path = os.path.join(base_dir, series_id, "metadata.json")
        with open(metadata_path) as json_file:
            json_data = json.load(json_file)
        
        num_slices = json_data["shape"][0]
        for slice_index in range(num_slices):
            entries_array[abs_slice_index] = (series_id, slice_index)
            abs_slice_index += 1
    
    return entries_array

def save_entries_array(entries_array, target_path, split):
    """
    Save the entries array to a .npy file.
    """
    file_path = os.path.join(target_path, f"entries-{split.upper()}.npy")
    np.save(file_path, entries_array)

def main(dataset_name):
    base_dir = os.path.join("datasets", dataset_name, "data")
    target_path = os.path.join("datasets", dataset_name, "extra")
    os.makedirs(target_path, exist_ok=True)

    series_ids = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    slice_counts = load_slice_counts(base_dir, series_ids)

    splits = {"train": 0.8, "val": 0.2}
    split_ids = create_subsets(series_ids, splits)

    for split in splits.keys():
        print(f"Processing subset: {split}")
        series_dirs = split_ids[split]
        entries_array = prepare_entries_array(series_dirs, slice_counts, base_dir)
        save_entries_array(entries_array, target_path, split)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)
