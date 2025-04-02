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
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    path_to_reports = os.path.join(
        data_path, "niftis/CT-RATE/dataset/radiology_text_reports/train_reports.csv"
    )
    output_path = os.path.join(project_path, "mllm/preprocessing/out")

    with open(
        "mllm/preprocessing/prompts/restructure.txt", mode="r", encoding="utf-8"
    ) as f:
        system_prompt = f.read()

    # important columns: "VolumeName", "Findings_EN"
    df = pd.read_csv(path_to_reports)

    # STEPS:
    # remove short reports (cutoff 400 chars)
    # get unique reports and asign an index to each
    # create mapping of image to unique report
    # save this mapping in a file in case script crashes
    # for each unique report: ask ollama to restructure it
    #     parse output: get text wrapped in <report></report>
    #     save restructured report in the output_path


if __name__ == "__main__":
    main()
