import os
import pandas as pd
import numpy as np
from scipy.special import expit
import argparse
import warnings

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModel
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
    hamming_loss,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


class RadBertClassifier(nn.Module):
    def __init__(self, n_classes=18):
        super().__init__()

        self.config = AutoConfig.from_pretrained("zzxslp/RadBERT-RoBERTa-4m")
        self.model = AutoModel.from_pretrained(
            "zzxslp/RadBERT-RoBERTa-4m", config=self.config
        )

        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)

        return output


class TextDataset(Dataset):
    def __init__(self, reports_df, tokenizer, max_len):
        self.reports_df = reports_df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reports_df)

    def __getitem__(self, idx):

        image_id, text = (
            self.reports_df.iloc[idx]["image"],
            self.reports_df.iloc[idx]["answer"],
        )

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image_id": image_id,
        }


def reports_to_labels(model, data_loader, device):
    model = model.eval()
    image_ids = []
    pred_logits = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)

            image_ids.extend(d["image_id"])
            logits = logits.detach().cpu().numpy()

            pred_logits.extend([logit for logit in logits])

    pred_logits = np.array(pred_logits)
    pred_labels = expit(pred_logits)

    pred_labels[pred_labels >= 0.5] = 1
    pred_labels[pred_labels < 0.5] = 0

    return image_ids, pred_labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports_path", type=str, default="runs/test_generate/evaluation"
    )
    parser.add_argument(
        "--classifier_model_path",
        type=str,
        default="mllm/evaluation/classifier/RadBertClassifier.pth",
    )
    parser.add_argument("--labels_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


def load_reports_folder(args):
    report_files = [x for x in os.listdir(args.reports_path) if x.endswith(".txt")]
    assert len(report_files) > 0, f"Found 0 reports in path {report_files}"

    rows = []
    for filename in report_files:
        mapid = filename.split(".txt")[0]
        filepath = os.path.join(args.reports_path, filename)

        with open(filepath, "r") as file:
            line = file.read()
            rows.append({"image": mapid, "answer": line})

    return pd.DataFrame(rows)


def load_reports_csv(args):

    reports = pd.read_csv(args.reports_path, sep=";")

    reports = reports.rename(columns={"VolumeName": "image", "report": "answer"})
    reports["image"] = reports["image"].apply(lambda x: x.replace(".nii.gz", ""))

    return reports


def load_labels(args):

    labels_df = pd.read_csv(args.labels_path)
    labels_df = labels_df.rename(columns={"VolumeName": "image"})
    labels_df["image"] = labels_df["image"].apply(lambda x: x.replace(".nii.gz", ""))

    return labels_df


def evaluate_model(y_true, y_pred, column_names):

    print("--- Evaluation Results ---")

    exact_match_ratio = accuracy_score(y_true, y_pred)
    print(f"Exact Match Ratio (Subset Accuracy): {exact_match_ratio:.4f}")
    print("    Proportion of samples where all labels were predicted correctly.\n")

    h_loss = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {h_loss:.4f}")
    print(
        "   Proportion of incorrect labels to the total number of labels. Lower is better.\n"
    )

    print("Per-Sample Averages (considers each sample's prediction quality):")
    precision_samples = precision_score(
        y_true, y_pred, average="samples", zero_division=0
    )
    recall_samples = recall_score(y_true, y_pred, average="samples", zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
    print(f"   Precision (samples): {precision_samples:.4f}")
    print(f"   Recall (samples):    {recall_samples:.4f}")
    print(f"   F1-Score (samples):  {f1_samples:.4f}\n")

    print("Micro-Averaged (global, considers all TPs, FPs, FNs):")
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"   Precision (micro): {precision_micro:.4f}")
    print(f"   Recall (micro):    {recall_micro:.4f}")
    print(f"   F1-Score (micro):  {f1_micro:.4f}\n")

    print("Macro-Averaged (unweighted average across labels):")
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"   Precision (macro): {precision_macro:.4f}")
    print(f"   Recall (macro):    {recall_macro:.4f}")
    print(f"   F1-Score (macro):  {f1_macro:.4f}")
    print(
        "   Note: Macro average can be heavily influenced by performance on infrequent labels.\n"
    )

    print("Weighted-Averaged (average across labels, weighted by label frequency):")
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"   Precision (weighted): {precision_weighted:.4f}")
    print(f"   Recall (weighted):    {recall_weighted:.4f}")
    print(f"   F1-Score (weighted):  {f1_weighted:.4f}\n")

    jaccard_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    jaccard_micro = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
    jaccard_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Jaccard Score (samples): {jaccard_samples:.4f}")
    print(f"   Jaccard Score (micro):   {jaccard_micro:.4f}")
    print(f"   Jaccard Score (macro):   {jaccard_macro:.4f}")
    print("    Similarity between true and predicted label sets. Higher is better.\n")

    # label_names = [f"Label_{i}" for i in range(y_true.shape[1])]
    report = classification_report(
        y_true, y_pred, target_names=column_names, zero_division=0
    )
    print("Classification Report (per label):\n", report)
    print("    Shows performance for each individual label.")
    print("   - support: The number of true instances for each label.")
    print("   - macro avg: Unweighted average of metrics across labels.")
    print("   - weighted avg: Average of metrics weighted by support for each label.")
    print(
        "   - samples avg: (Only for multi-label) Average of metrics for each sample.\n"
    )

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    print("Matrix Confusion Matrices (per label):")
    for i, matrix in enumerate(mcm):
        print(f"\nConfusion Matrix for {column_names[i]}:")
        print(f"  [[TN={matrix[0, 0]} FP={matrix[0, 1]}]")
        print(f"   [FN={matrix[1, 0]} TP={matrix[1, 1]}]]")


def main():

    args = get_args()

    reports_df = load_reports_csv(args)

    labels_df = load_labels(args)
    label_columns = labels_df.columns[1:]

    tokenizer = AutoTokenizer.from_pretrained(
        "zzxslp/RadBERT-RoBERTa-4m", do_lower_case=True
    )

    reports_dataset = TextDataset(
        reports_df=reports_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    reports_loader = DataLoader(
        reports_dataset, batch_size=args.batch_size, shuffle=False
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        warnings.warn("Cuda not available. Using CPU.")
        device = torch.device("cpu")

    model = RadBertClassifier(n_classes=len(label_columns))

    state_dict = torch.load(args.classifier_model_path)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    image_ids, predicted_labels = reports_to_labels(model, reports_loader, device)

    predicted_labels = pd.DataFrame(predicted_labels, columns=label_columns)
    predicted_labels["image"] = image_ids

    labels_df = labels_df.set_index("image")
    predicted_labels = predicted_labels.set_index("image")

    y_true = labels_df.loc[predicted_labels.index].values
    y_pred = predicted_labels.loc[predicted_labels.index].values

    evaluate_model(y_true, y_pred, label_columns)


if __name__ == "__main__":
    main()
