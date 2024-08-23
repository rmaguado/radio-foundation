import os
import json
from collections import OrderedDict
from tqdm import tqdm
import random
import sys


def assign_subsets(data, proportions):
    """
    Assign each series into a subsets based on the given proportions.
    """
    random.shuffle(data)
    assignments = {}
    num_items = len(data)
    start_index = 0
    for subset_name, proportion in proportions.items():
        subset_size = int(proportion * num_items)
        end_index = start_index + subset_size
        assignments[subset_name] = data[start_index:end_index]
        start_index = end_index
    
    if start_index < num_items:
        assignments[list(proportions.keys())[0]] += data[start_index:]
        
    for subset in assignments.values():
        subset.sort()
    
    return assignments

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

def compile_entries(subsets, slice_counts):
    entries = {}
    
    for subset_name, series_ids in subsets.items():
        entry = OrderedDict([
            (s_id, {
                "slices": slice_counts[s_id]
            }) for s_id in series_ids
        ])
        entries[subset_name] = entry
    return entries
    
def save_entries(target_path, entries_data):
    for subset_name, entry in entries_data.items():
        with open(os.path.join(target_path, f"{subset_name}.json"), "w") as f:
            json.dump(entry, f, allow_nan=False, indent="    ")

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
    subsets = assign_subsets(series_ids, splits)    
    entries = compile_entries(subsets, slice_counts)
    
    save_entries(target_path, entries)

if __name__ == "__main__":
    random.seed(0)
    dataset_name = sys.argv[1]
    main(dataset_name)
