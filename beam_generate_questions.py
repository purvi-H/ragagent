import torch
import transformers
import csv
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
from torch import cuda
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch

print("entered the file")

def setup_huggingface():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_auth = "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"
    login(token=hf_auth)

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

    pretrained_model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, config=model_config)

    return pretrained_model, tokenizer

def setup_pipeline(pretrained_model, tokenizer):
    llama_pipeline = pipeline(
        task="text-generation",
        model=pretrained_model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer,
        return_full_text=True,
        do_sample=True,
        num_beams=4,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.01,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return llama_pipeline

def setup_sharded_model(pretrained_model):
    accelerator = Accelerator()
    save_directory = "/exports/eddie/scratch/s2024596/ragagent/sharded_model"
    accelerator.save_model(model=pretrained_model, save_directory=save_directory, max_shard_size="2GB")

    device_map = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    sharded_model = load_checkpoint_and_dispatch(pretrained_model, checkpoint=save_directory, device_map='auto', no_split_module_classes=['Block'])

    return sharded_model

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest question extracter assistant. Always answer as helpfully as possible, while being safe. Keep answers relevant to the data provided."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline, verbose = True)
    print("llm setup")

    chart_data_directory = "/home/s2024596/ragagent/dataset"
    with open("/home/s2024596/ragagent/dataset/new_output.txt", "r") as output_file:
        lines = output_file.read().splitlines()[:3]

        for i, line in enumerate(lines):
            split = line.split("|")
            chart_type = split[0].strip()
            chart_data_filename = split[1].strip().replace('.txt', '.csv')

            if chart_type == "Simple":
                chart_data_path = os.path.join(chart_data_directory, "data", chart_data_filename)
            else: # if it is "Complex"
                chart_data_path = os.path.join(chart_data_directory, "multiColumn/data", chart_data_filename)
            with open(chart_data_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                chart_data = list(reader)

            with open('/home/s2024596/ragagent/dataset/new_charttitle.txt', 'r') as file:
                chart_title_lines = file.readlines()
                chart_title = chart_title_lines[i].strip()

            with open('/home/s2024596/ragagent/dataset/new_charttype.txt', 'r') as file:
                chart_type_lines = file.readlines()
                chart_type = chart_type_lines[i].strip()

            instruction = """
            Question: {question}
            Answer in this format: 
            1..	
            2.. 
            3..	
            4.. 
            5..	
            """
            prompt_template = get_prompt(instruction)

            chain = PromptTemplate.from_template(prompt_template) | llm
            question = "Generate 5 questions that can be answered only from this data about {chart_title} : {chart_data}"
            question = question.format(chart_title=chart_title, chart_data=chart_data)
            
#     question = """Generate 5 questions that can be answered only from this data about Global spending on motorsports sponsorships 2011 to 2017:
# [["Year", ["2017", "2016", "2015", "2014", "2013", "2012", "2011"]], ["Spending in billion U.S. dollars", ["5.75", "5.58", "5.43", "5.26", "5.12", "4.97", "4.83"]]]"""
    
            print(chain.invoke({"question": question}))
            print("generating result from llm")

