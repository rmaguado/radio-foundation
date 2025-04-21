import os
import pandas as pd
from dotenv import load_dotenv
from ollama import Client
import logging
import concurrent.futures
import time


with open("prompts/parse_claims.txt", "r") as f:
    PARSE_CLAIMS_PROMPT = f.read()


def get_response(system_prompt, query, client, model="llama3.3:latest"):
    ollama_response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return ollama_response["message"]["content"]


def parse_claims(report_id, report, output_file, client):
    """
    Parses the claims from the report using the specified system prompt.
    """
    try:
        response = get_response(PARSE_CLAIMS_PROMPT, report, client)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)

    except Exception as e:
        logging.error(f"Error processing report {report_id}: {e}")
    time.sleep(0.2)


def parse_ground_truths(hosts):
    clients = [Client(host=host) for host in hosts]

    logging.basicConfig(
        filename="mllm/evaluation/out/restructure_reports.log", level=logging.DEBUG
    )

    reports_file = "mllm/preprocessing/out/valid/combined_reports.csv"
    mapping_file = "mllm/preprocessing/out/valid/report_mapping.csv"
    output_path = "mllm/evaluation/out/ground_truths"

    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(reports_file, delimiter=";")
    mapping_df = pd.read_csv(mapping_file)
    df = df.merge(mapping_df, on="VolumeName", how="left")

    processed_ids = set()
    for report_id in df["report_id"].unique():
        if os.path.exists(os.path.join(output_path, report_id + ".csv")):
            processed_ids.add(report_id)

    df = df[~df["report_id"].isin(processed_ids)].drop_duplicates(subset=["report_id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = []
        for i, (_, row) in enumerate(df.iterrows()):
            client = clients[i % len(hosts)]
            report = row["Findings_EN"]
            report_id = row["report_id"]
            output_file = os.path.join(output_path, report_id + ".csv")
            futures.append(
                executor.submit(parse_claims, report_id, report, output_file, client)
            )
        concurrent.futures.wait(futures)

    logging.info("Processing complete. Results saved.")


if __name__ == "__main__":
    ports_cc2 = list(range(11434, 11434 + 8))
    ports_cc3 = list(range(11434, 11434 + 4))
    hosts_cc2 = [f"http://127.0.0.1:{port}" for port in ports_cc2]
    hosts_cc3 = [f"http://192.168.36.203:{port}" for port in ports_cc3]
    parse_ground_truths(hosts_cc2 + hosts_cc3)
