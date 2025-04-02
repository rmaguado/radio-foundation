import os
from dotenv import load_dotenv
import pandas as pd
import ollama
from ollama import Client
import logging
import time
import concurrent.futures


def get_response(system_prompt, query, client, model="llama3.3:latest"):
    ollama_response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return ollama_response["message"]["content"]


def process_report(row, system_prompt, output_file, client):
    query = row["Findings_EN"]
    try:
        response = get_response(system_prompt, query, client)
        start_tag, end_tag = "<report>", "</report>"
        structured_report = (
            response.split(start_tag)[1].split(end_tag)[0].strip()
            if start_tag in response and end_tag in response
            else response
        )
        structured_report = structured_report.replace("\n", "").replace(";", ",")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{row['report_id']};{structured_report}\n")
    except Exception as e:
        logging.error(f"Error processing report {row['report_id']}: {e}")
    time.sleep(0.2)


def main(hosts):
    clients = [Client(host=host) for host in hosts]

    logging.basicConfig(
        filename="mllm/preprocessing/out/restructure_reports.log", level=logging.DEBUG
    )
    load_dotenv()
    project_path, data_path = os.getenv("PROJECTPATH"), os.getenv("DATAPATH")
    reports_file = os.path.join(
        data_path, "niftis/CT-RATE/dataset/radiology_text_reports/train_reports.csv"
    )
    output_path = os.path.join(project_path, "mllm/preprocessing/out")
    mapping_file = os.path.join(output_path, "report_mapping.csv")

    os.makedirs(output_path, exist_ok=True)
    with open("mllm/preprocessing/prompts/restructure.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    df = pd.read_csv(reports_file)[["VolumeName", "Findings_EN"]]
    df = df[df["Findings_EN"].str.len() >= 400]

    if os.path.exists(mapping_file):
        mapping_df = pd.read_csv(mapping_file)
        df = df.merge(mapping_df, on="VolumeName", how="left")
    else:
        unique_reports = df.drop_duplicates(subset="Findings_EN").reset_index(drop=True)
        unique_reports["report_id"] = unique_reports.index
        df = df.merge(unique_reports, on="Findings_EN", how="left")
        df[["VolumeName", "report_id"]].to_csv(mapping_file, index=False)

    processed_ids = set()
    output_files = [
        os.path.join(output_path, f"restructured_reports_{i}.csv")
        for i in range(len(clients))
    ]

    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                processed_ids.update(int(line.split(";")[0]) for line in f)

    df = df[~df["report_id"].isin(processed_ids)].drop_duplicates(subset=["report_id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = []
        for i, (_, row) in enumerate(df.iterrows()):
            output_file = output_files[i % len(output_files)]
            client = clients[i % len(hosts)]
            futures.append(
                executor.submit(process_report, row, system_prompt, output_file, client)
            )
        concurrent.futures.wait(futures)

    logging.info("Processing complete. Results saved.")


if __name__ == "__main__":
    ports_cc2 = list(range(11434, 11434 + 8))
    ports_cc3 = list(range(11434, 11434 + 4))
    hosts_cc2 = [f"http://127.0.0.1:{port}" for port in ports_cc2]
    hosts_cc3 = [f"http://192.168.36.203:{port}" for port in ports_cc3]
    main(hosts_cc2 + hosts_cc3)
