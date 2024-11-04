import os
import json
import torch

from dinov2.data.datasets import CtDataset


class NsclcRadiomicsEvalSex(CtDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_targets=True)
        self.series_targets = dict()
        self.extend_entries()

    def extend_entries(self):
        entries = self._get_entries()
        series_ids = list(set([x[0] for x in entries]))

        for s_id in series_ids:
            path_to_meta = os.path.join(
                "../datasets/NSCLC-Radiomics/data/", s_id, "metadata.json"
            )
            with open(path_to_meta, "rb") as f:
                metadata = json.load(f)
                self.series_targets[s_id] = ["M", "F"].index(metadata["other"]["sex"])

    def get_target(self, index: int):
        entries = self._get_entries()
        series_id = entries[index][0]
        return torch.tensor(self.series_targets[series_id])
