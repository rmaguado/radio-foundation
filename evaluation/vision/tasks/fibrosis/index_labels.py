import os
import pandas as pd

dataset = "australia"

died_0_1 = pd.read_csv(os.path.join(dataset, "mortality_0to1_died.csv"))["ID"].to_list()
survived_0_1 = pd.read_csv(os.path.join(dataset, "mortality_0to1_survived.csv"))[
    "ID"
].to_list()
died_1_3 = pd.read_csv(os.path.join(dataset, "mortality_1to3_died.csv"))["ID"].to_list()
survived_1_3 = pd.read_csv(os.path.join(dataset, "mortality_1to3_survived.csv"))[
    "ID"
].to_list()
died_3_5 = pd.read_csv(os.path.join(dataset, "mortality_3to5_died.csv"))["ID"].to_list()
survived_3_5 = pd.read_csv(os.path.join(dataset, "mortality_3to5_survived.csv"))[
    "ID"
].to_list()
survived_5_more = pd.read_csv(os.path.join(dataset, "mortality_5more.csv"))[
    "ID"
].to_list()

died_ids = died_0_1 + survived_0_1 + died_1_3
survived_ids = survived_1_3 + died_3_5 + survived_3_5 + survived_5_more


with open("labels_australia.csv", "a") as f:
    f.write("mapid,label\n")
    for name in died_ids:
        f.write(f"{name},0\n")
    for name in survived_ids:
        f.write(f"{name},1\n")
