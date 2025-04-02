import os
from dotenv import load_dotenv
import pandas as pd
import ollama
import logging
import time
from tqdm import tqdm


def get_response(system_prompt, query, model="llama3.3:latest"):

    ollama_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    return ollama_response["message"]["content"]


def main():
    logger = configure_logging()
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    path_to_reports = os.path.join(
        data_path, "niftis/CT-RATE/dataset/radiology_text_reports/train_reports.csv"
    )
    output_path = os.path.join(project_path, "mllm/preprocessing/out")
    mapping_file = os.path.join(output_path, "report_mapping.csv")
    restructured_reports_file = os.path.join(output_path, "restructured_reports.csv")

    os.makedirs(output_path, exist_ok=True)

    with open(
        "mllm/preprocessing/prompts/restructure.txt", mode="r", encoding="utf-8"
    ) as f:
        system_prompt = f.read()

    df = pd.read_csv(path_to_reports)[["VolumeName", "Findings_EN"]]
    df = df[df["Findings_EN"].str.len() >= 400]

    if os.path.exists(mapping_file):
        logger.info("Found existing mapping file.")
        mapping_df = pd.read_csv(mapping_file)
        df = df.merge(mapping_df, on="VolumeName", how="left")
    else:
        logger.info("Creating new mapping file.")
        unique_reports = df.drop_duplicates(subset="Findings_EN").reset_index(drop=True)
        unique_reports["report_id"] = unique_reports.index
        df = df.merge(unique_reports, on="Findings_EN", how="left")
        df["VolumeName"] = df["VolumeName_x"]
        df[["VolumeName", "report_id"]].to_csv(mapping_file, index=False)

    processed_ids = set()
    if os.path.exists(restructured_reports_file):
        with open(restructured_reports_file, "r", encoding="utf-8") as f:
            processed_ids = {int(line.split(",")[0]) for line in f}

    for _, row in tqdm(
        df.drop_duplicates(subset=["report_id"]).iterrows(),
        total=len(df["report_id"].unique()),
        desc="Processing reports",
    ):
        if row["report_id"] in processed_ids:
            continue

        query = row["Findings_EN"]

        try:
            response = get_response(system_prompt, query)

            start_tag = "<report>"
            end_tag = "</report>"
            if start_tag in response and end_tag in response:
                structured_report = (
                    response.split(start_tag)[1].split(end_tag)[0].strip()
                )
            else:
                structured_report = response

            structured_report = structured_report.replace("\n", "").replace(",", ";")

            with open(restructured_reports_file, "a", encoding="utf-8") as f:
                f.write(f"{row['report_id']},{structured_report}\n")

            logger.info(f"Processed report {row['report_id']} successfully.")

        except Exception as e:
            logger.error(f"Error processing report {row['report_id']}: {e}")

        time.sleep(0.05)

    logger.info("Processing complete. Results saved.")


def configure_logging():

    logging.basicConfig(
        filename="mllm/preprocessing/out/restructure_reports.log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )

    return logging.getLogger("ReportOLlama")


if __name__ == "__main__":
    main()
