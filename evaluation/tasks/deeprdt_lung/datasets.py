from typing import Tuple, Any
import pandas as pd
import torch


class DeepRDT_lung:
    def __init__(
        self,
        embeddings_provider: Any,
        metadata_path: str,
        label: str,
        true_category: str,
    ):
        self.df = pd.read_csv(metadata_path)
        self.label = label
        self.true_category = true_category.strip().lower()
        self.embeddings_provider = embeddings_provider
        self.map_ids = self.embeddings_provider.map_ids
        self.targets = self.index_targets()

    def is_valid_value(self, val: Any) -> bool:
        """Check if value is valid (not missing/blank/na/nan)"""
        if pd.isna(val):
            return False
        val_str = str(val).strip().lower()
        return val_str not in {"", "na", "nan", "none"}

    def index_targets(self):
        if self.label not in self.df.columns:
            raise ValueError(f"Label '{self.label}' not found in metadata.")

        df_filtered = self.df[self.df["MAPID"].isin(self.map_ids)]

        label_series = df_filtered[self.label].astype(str).str.strip().str.lower()

        # Apply validity mask
        valid_mask = label_series.apply(self.is_valid_value)
        valid_values = label_series[valid_mask].unique()

        print("Categories:", valid_values)

        if len(valid_values) != 2:
            raise ValueError(
                f"Label '{self.label}' must have exactly 2 unique non-missing values. Found: {valid_values}"
            )

        if self.true_category not in valid_values:
            raise ValueError(
                f"Provided true_category '{self.true_category}' not found in label values: {valid_values}"
            )

        def target_fn(val: Any) -> Any:
            val_str = str(val).strip().lower()
            if not self.is_valid_value(val_str):
                return None
            return val_str == self.true_category

        target_column = df_filtered.set_index("MAPID")[self.label]
        targets = {
            map_id: target_fn(target_column[map_id])
            for map_id in self.map_ids
            if map_id in target_column
        }

        targets = {k: v for k, v in targets.items() if v is not None}
        self.map_ids = [k for k in self.map_ids if k in targets]

        return targets

    def get_target(self, index):
        return self.targets[self.map_ids[index]]

    def __len__(self) -> int:
        return len(self.map_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        map_id = self.map_ids[index]
        label = self.get_target(index)
        embeddings = self.embeddings_provider.get_embeddings(map_id)
        return embeddings, label
