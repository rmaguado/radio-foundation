import os
import numpy as np
import json
from tqdm import tqdm
import random


def get_series_ids(dataset_path):
    series_ids = [
        d for d in os.listdir(dataset_path) \
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    return series_ids

def count_slices(dataset_path, series_ids):
    slice_counts = dict()
    for series_id in tqdm(series_ids):
        metadata_path = os.path.join(dataset_path, series_id, "metadata.json")
        with open(metadata_path) as json_file:
            json_data = json.load(json_file)
        slice_counts[series_id] = json_data["shape"][0]
    return slice_counts

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


dtype = np.dtype(
    [
        ("dataset", "U256"),
        ("series_id", "U256"),
        ("slice_index", "uint16"),
    ]
)

dataset_names = ["LIDC-IDRI", "NSCLC-Radiomics", "NSCLC-Radiogenomics"]
target_path = "extra"
splits = {
    "train": 0.8,
    "test": 0.2,
}

split_datasets = dict()
slice_counts_datasets = dict()

os.makedirs(target_path, exist_ok=True)
for dataset in dataset_names:
    path_to_data = os.path.join("..", dataset, "data")
    series_ids = get_series_ids(path_to_data)
    slice_counts_datasets[dataset] = count_slices(path_to_data, series_ids)
    
    split_datasets[dataset] = create_subsets(series_ids, splits)
    
    print(dataset)
    for split in splits.keys():
        print(split, sum([slice_counts_datasets[dataset][x] for x in split_datasets[dataset][split]]))


for split in splits.keys():

    print(split)
    split_series_ids_datasets = {dataset:split_datasets[dataset][split] for dataset in dataset_names}
    length = sum([
        slice_counts_datasets[dataset][x] \
        for dataset in dataset_names \
        for x in split_datasets[dataset][split]
        
    ])
    print(length)

    entries_array = np.empty(length, dtype=dtype)

    abs_index = 0
    for dataset in dataset_names:
        for series_id in split_series_ids_datasets[dataset]:
            num_slices = slice_counts_datasets[dataset][series_id]

            for slice_index in range(num_slices):
                entries_array[abs_index] = (
                    dataset,
                    series_id,
                    slice_index
                )
                abs_index += 1

    extra_full_path = os.path.join(target_path, f"entries-{split.upper()}.npy")
    np.save(extra_full_path, entries_array)
