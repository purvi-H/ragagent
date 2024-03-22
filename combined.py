import os
import csv
import torch
import transformers
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch
from langchain.document_loaders.csv_loader import CSVLoader
from torch import cuda
import combined_generate_qs
from langchain_experimental.llms import JsonFormer

'''
generate qs needs csv and the title
'''

set_seed(42)

def parse_rows(chart_data_directory, i, line):
    split = line.split("|")
    chart_type = split[0].strip()
    chart_data_filename = split[1].strip().replace('.txt', '.csv')

    # chart_data -> the csv file
    if chart_type == "Simple":
        chart_data_path = os.path.join(chart_data_directory, "data", chart_data_filename)
    else: # if it is "Complex"
        chart_data_path = os.path.join(chart_data_directory, "multiColumn/data", chart_data_filename)
    with open(chart_data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        chart_data = '\n'.join(','.join(row) for row in reader)
        # csv_data = [row for row in reader]
        # json_data = json.dumps(csv_data, indent=1)

    
    # chart_title -> title of the chart
    with open('/home/s2024596/ragagent/dataset/new_charttitle.txt', 'r') as file:
        chart_title_lines = file.readlines()
        chart_title = chart_title_lines[i].strip()

    # chart_data -> bar or line
    with open('/home/s2024596/ragagent/dataset/new_charttype.txt', 'r') as file:
        chart_type_lines = file.readlines()
        chart_type = chart_type_lines[i].strip()

    return chart_type, chart_data, chart_title


if __name__ == "__main__":

    # json schema for the questions
    json_schemaa = {
                    "type": "object",
                    "properties": {
                        "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 5,
                        "maxItems": 5
                        }
                    },
                    "required": ["questions"]
                    }
    # for each row in the file, extracting the relevant chart info
    chart_data_directory = "/home/s2024596/ragagent/dataset"
    with open("/home/s2024596/ragagent/dataset/new_output.txt", "r") as output_file:
        lines = output_file.read().splitlines()[:2]
        for i, line in enumerate(lines):
            print(line)
            chart_type, chart_data, chart_title =  parse_rows(chart_data_directory, i, line)
            
            # GENERATE QUESTIONS
            pretrained_model, tokenizer = combined_generate_qs.setup_huggingface()
            sharded_model = combined_generate_qs.setup_sharded_model(pretrained_model)
            llama_pipeline = combined_generate_qs.setup_pipeline(sharded_model, tokenizer)
            llm = combined_generate_qs.HuggingFacePipeline(pipeline=llama_pipeline, verbose=True)
            print("llm setup")

            instruction = """
            Generate 5 questions that can be answered only from this data about {chart_title} : {chart_data}
            Generate the questions based on the following schema:
            """
            instruction = instruction.format(chart_title=chart_title, chart_data=chart_data)
            prompt_template = combined_generate_qs.get_prompt(instruction)
            updated_prompt_template = combined_generate_qs.PromptTemplate.from_template(prompt_template)

            jsonformer = JsonFormer(pipeline = llm, json_schema = json_schemaa)
            print("generating questions from llm")
            print("")
            print("")
            results = jsonformer.invoke(prompt_template)
            print(results)
