import os
from dotenv import load_dotenv
import pandas as pd
import ollama
import re
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


def parse_qa(text):
    pattern = re.compile(r"(<q>.*?</q><a>.*?</a>)")
    matches = pattern.findall(text)
    return matches


def generate_conversation_open(radiology_report):

    system_prompt = """You are a dataset generator creating an instruction dataset for multimodal instruction tuning. Given a radiology report from a CT scan, generate user questions and accurate answers strictly based on the report. Each question must be open-ended and avoid assuming the presence or absence of specific abnormalities. Each question must be standalone. Questions should not make direct reference to the report. Use the following format and ensure it is replicated exactly: {"question": "[QUESTION]", "answer": "[ANSWER]"}. Ensure full utilization of the report's information. """

    response = get_response(system_prompt, radiology_report)

    return response


def generate_conversation_closed(radiology_report):

    system_prompt = """You are a dataset generator creating an instruction dataset for multimodal instruction tuning. Given a radiology report from a CT scan, generate user questions and accurate answers strictly based on the report. Each question must be standalone and not make direct reference to the report. Use the following format and ensure it is replicated exactly: {"question": "[QUESTION]", "answer": "[ANSWER]"}. Ensure full utilization of the report's information. """

    response = get_response(system_prompt, radiology_report)

    return response


def generate_open_questions(radiology_report):
    system_prompt = """You are a dataset generator creating an instruction dataset for multimodal instruction tuning. Given a radiology report from a CT scan, generate open-ended user questions that encourage general descriptions, summaries, or key findings of the scan. Ensure that answers are accurate and fully grounded in the report without referencing it directly. Use the following format and ensure it is replicated exactly:{"question": "[QUESTION]", "answer": "[ANSWER]"}. """

    response = get_response(system_prompt, radiology_report)

    return response


def augment_instructions(question_answer):
    system_prompt = "You are a dataset generator creating augmented question-answer pairs for multimodal instruction tuning. Given a CT scan-related Q&A pair, rephrase the question in different ways (e.g., as a command or with alternative wording) while ensuring factual consistency. Adjust the answer if needed. Format: <q>Question 1</q><a>Answer 1</a>."

    response = get_response(system_prompt, question_answer)

    return response


def label_reports():
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    path_to_reports = os.path.join(
        data_path, "niftis/CT-RATE/dataset/radiology_text_reports/train_reports.csv"
    )

    output_path = os.path.join(project_path, "vqa/data/train")
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(path_to_reports)

    volume_names = df["VolumeName"]
    volume_names = list(set(["_".join(name.split("_")[:3]) for name in volume_names]))

    reports = df["Findings_EN"].unique()

    print(f"Found {len(reports)} unique reports.")

    for volume_name in tqdm(volume_names[:10]):
        report = df[df["VolumeName"] == volume_name + "_1.nii.gz"]["Findings_EN"].iloc[
            0
        ]

        report_path = os.path.join(output_path, volume_name)
        os.makedirs(report_path, exist_ok=True)

        with open(f"{report_path}/report.txt", "w", encoding="utf-8") as f:
            f.write(report)


def generate_report_qa():
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    report_data_path = os.path.join(project_path, "vqa/data/train")
    report_names = os.listdir(report_data_path)

    for report_name in tqdm(report_names):
        data_path = os.path.join(report_data_path, report_name)
        with open(f"{data_path}/report.txt", "r", encoding="utf-8") as f:
            report_text = f.read()

        response_open = generate_conversation_open(report_text)
        response_closed = generate_conversation_closed(report_text)

        with open(f"{data_path}/conversation.txt", "w", encoding="utf-8") as f:
            f.write(response_open + "\n" + response_closed)


def main():
    # label_reports()
    generate_report_qa()


if __name__ == "__main__":
    main()
