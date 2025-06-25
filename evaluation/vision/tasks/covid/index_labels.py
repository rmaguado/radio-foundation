import os
import glob

path_to_data = "/home/48078029W/data/niftis/ccccii-cleaned_new/**/*.nii.gz"

files = glob.glob(path_to_data, recursive=True)

Normal_names = [
    file.split("/")[-1].split(".nii")[0] for file in files if "Normal" in file
]
NCP_names = [file.split("/")[-1].split(".nii")[0] for file in files if "NCP" in file]
CP_names = [
    file.split("/")[-1].split(".nii")[0]
    for file in files
    if "CP" in file and "NCP" not in file
]


# remove duplicates
Normal_names = [x for x in Normal_names if x not in NCP_names and x not in CP_names]
NCP_names = [x for x in NCP_names if x not in CP_names and x not in Normal_names]
CP_names = [x for x in CP_names if x not in NCP_names and x not in Normal_names]


with open("labels_covid.csv", "a") as f:
    f.write("mapid,label\n")
    for name in NCP_names:
        f.write(f"{name},0\n")
    for name in CP_names:
        f.write(f"{name},1\n")

with open("labels_pneumonia.csv", "a") as f:
    f.write("mapid,label\n")
    for name in Normal_names:
        f.write(f"{name},0\n")
    for name in NCP_names:
        f.write(f"{name},1\n")
    for name in CP_names:
        f.write(f"{name},1\n")
