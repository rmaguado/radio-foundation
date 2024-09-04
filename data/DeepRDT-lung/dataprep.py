import os
from os.path import basename, normpath
import pandas as pd
import SimpleITK as sitk
import pydicom
from tqdm import tqdm

from data.dataset import DatasetBase, validate_ct_dicom, get_argpase


class DeepRDTlung(DatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def get_series_paths(self):

        df_path = self.config["df_path"]
        patient_ids_list = [
            x.split("_")[-1].split(".")[0]
            for x in os.listdir(df_path)
            if x.endswith(".csv")
        ]

        datapath = self.config["dataset_path"]
        series_paths = []
        reader = sitk.ImageSeriesReader()

        print("Walking dataset directories.")
        total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(datapath))
        for data_folder, dirs, files in tqdm(os.walk(datapath), total=total_dirs):
            series_ids = reader.GetGDCMSeriesIDs(data_folder)
            if not series_ids:
                continue

            patient_id = basename(normpath(data_folder))
            if patient_id not in patient_ids_list:
                continue
            df = pd.read_csv(os.path.join(df_path, f"ct_df_filtered_{patient_id}.csv"))
            dosisplan_series_id = df["SeriesInstanceUID"].iloc[0]

            series_file_names = reader.GetGDCMSeriesFileNames(
                data_folder, dosisplan_series_id
            )
            if not series_file_names:
                continue

            first_file = series_file_names[0]
            dcm = pydicom.dcmread(first_file, stop_before_pixels=True)
            if validate_ct_dicom(dcm, data_folder):
                series_paths.append((dosisplan_series_id, data_folder))

        return series_paths


def main(dataset_name: str, root_path: str, db_path: str):
    dataset_path = os.path.join(root_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "target_path": db_path,
        "df_path": "/home/rmaguado/cuda/AI/alba/deepRDT_lung/data/dfs",
    }
    dataset = DeepRDTlung(config)
    dataset.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args.dataset_name, args.root_path, args.db_path)
