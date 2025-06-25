import os
import pandas as pd
from dotenv import load_dotenv
from ollama import Client
import logging
import concurrent.futures
import time
import json
import re
from pathlib import Path

with open("mllm/evaluation/prompts/parse_claims.txt", "r") as f:
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


def extract_json_from_report(text):
    """Extract JSON content between <output> tags"""
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
    return None


def process_single_report(report_id, report_text, client):
    try:
        response = get_response(PARSE_CLAIMS_PROMPT, report_text, client)
        result = extract_json_from_report(response)

        if result:
            result["report_id"] = report_id
            # Convert boolean values to strings for consistent CSV output
            for key in result:
                if isinstance(result[key], bool):
                    result[key] = str(result[key]).lower()
            return result
        else:
            logging.warning(f"Failed to parse response for report {report_id}")
            return None

    except Exception as e:
        logging.error(f"Error processing report {report_id}: {str(e)}")
        return None


def parse_reports(
    reports, hosts, output_path="mllm/evaluation/out", checkpoint_interval=10
):
    """Parse multiple reports using multiple Ollama hosts in parallel with checkpointing"""

    clients = [Client(host=host) for host in hosts]

    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_path, "restructure_reports.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Define the expected findings (columns for our CSV)
    findings = [
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]

    # Check for existing checkpoint
    checkpoint_file = os.path.join(output_path, "checkpoint.json")
    processed_reports = set()
    results = []

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                processed_reports = set(checkpoint_data["processed_reports"])
                results = checkpoint_data["results"]
                logging.info(
                    f"Resuming from checkpoint. {len(processed_reports)} reports already processed."
                )
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting fresh.")

    # Filter out already processed reports
    reports_to_process = {
        k: v for k, v in reports.items() if k not in processed_reports
    }

    if not reports_to_process:
        logging.info("All reports already processed.")
    else:
        start_time = time.time()
        processed_count = 0

        try:
            # Process reports in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(clients)
            ) as executor:
                futures = []
                for report_id, report_text in reports_to_process.items():
                    # Round-robin client assignment
                    client = clients[len(futures) % len(clients)]
                    futures.append(
                        executor.submit(
                            process_single_report, report_id, report_text, client
                        )
                    )

                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    result = future.result()
                    if result:
                        results.append(result)
                        processed_reports.add(result["report_id"])

                    # Save checkpoint periodically
                    if i % checkpoint_interval == 0:
                        save_checkpoint(output_path, processed_reports, results)
                        logging.info(f"Checkpoint saved after {i} reports")

                # Final checkpoint save
                save_checkpoint(output_path, processed_reports, results)

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            save_checkpoint(output_path, processed_reports, results)
            raise

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)

        # Ensure all findings are present as columns
        for finding in findings:
            if finding not in df.columns:
                df[finding] = "false"

        # Reorder columns with report_id first
        columns = ["report_id"] + findings
        df = df[columns]

        # Save final results
        output_file = os.path.join(output_path, "ct_findings_classification.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Final results saved to {output_file}")

    else:
        logging.warning("No valid results were generated")

    if reports_to_process:
        elapsed = time.time() - start_time
        logging.info(f"Processing complete. Time elapsed: {elapsed:.2f} seconds")
    return df if results else None


def save_checkpoint(output_path, processed_reports, results):
    """Save progress checkpoint"""
    checkpoint_file = os.path.join(output_path, "checkpoint.json")
    try:
        with open(checkpoint_file, "w") as f:
            json.dump(
                {"processed_reports": list(processed_reports), "results": results}, f
            )
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")


def load_true_reports():
    true_reports_path = (
        "mllm/preprocessing/out/abnormalities/valid/combined_reports.csv"
    )

    true_reports_df = pd.read_csv(true_reports_path, sep=";")
    true_reports_df["base_volume_name"] = true_reports_df["VolumeName"].apply(
        lambda x: (
            "_".join(x.split("_")[:-1])
            if x.endswith(".nii.gz")
            and x.split("_")[-1].replace(".nii.gz", "").isdigit()
            else x.replace(".nii.gz", "")
        )
    )
    true_reports_df = true_reports_df.drop_duplicates(subset=["base_volume_name"])
    true_reports_df = true_reports_df.set_index("base_volume_name")["report"]

    return true_reports_df.to_dict()


def load_generated_reports(run_name):

    reports_folder_path = f"runs/{run_name}/evaluation"

    reports_map = {}

    filenames = os.listdir(reports_folder_path)
    for file in filenames:
        base_name_with_ext = "_".join(file.replace(".txt", "").split("_")[:-1])
        base_name = base_name_with_ext

        report_path = os.path.join(reports_folder_path, file)

        with open(report_path, "r") as f:
            report = f.read()

        if base_name not in reports_map:
            reports_map[base_name] = []
        reports_map[base_name].append(report)

    combined_reports = {}

    for base_name in sorted(reports_map.keys()):

        combined_reports[base_name] = " ".join(reports_map[base_name])

    return combined_reports


if __name__ == "__main__":
    ports_cc2 = list(range(11434, 11434 + 6))
    ports_cc3 = list(range(11434, 11434 + 6))
    hosts_cc2 = [f"http://127.0.0.1:{port}" for port in ports_cc2]
    hosts_cc3 = [f"http://192.168.36.203:{port}" for port in ports_cc3]
    hosts = hosts_cc2 + hosts_cc3  # ["http://127.0.0.1:11434"]

    run_name = "cls_frozen_onestep_gen"
    output_path = f"mllm/evaluation/out/{run_name}"
    reports = load_generated_reports(run_name)
    parse_reports(reports, hosts, output_path=output_path)

    run_name = "ground_truth"
    output_path = f"mllm/evaluation/out/{run_name}"
    reports = load_true_reports()
    parse_reports(reports, hosts, output_path=output_path)

    for run_name in [
        "cls_frozen_onestep_ct_rate_gen",
        "ct_clip_onestep_gen",
        "ct_fm_onestep_gen",
        "cls_frozen_onestep_large_gen",
    ]:

        output_path = f"mllm/evaluation/out/{run_name}"
        reports = load_generated_reports(run_name)
        parse_reports(reports, hosts, output_path=output_path)
