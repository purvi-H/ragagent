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
        # return_full_text=True,
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
You are a helpful, respectful and honest question extracter assistant. Always answer as helpfully as possible, while being safe. 
Here is an exmaple of how to answer a question: 

Question: Generate 5 short questions that can be answered only from this data: [['Country', 'Rate per 100 inhabitants'], ['Africa', '0.4'], ['Arab States', '8.1'], ['Asia & Pacific', '14.4'], ['World', '14.9'], ['CIS', '19.8'], ['The Americas', '22.0'], ['Europe', '31.9']]. Include the data point about which the question is in the question.

Answer in this format:
1..	
2..
3..	
4..
5..

Think step by step:

Step 1: Extract all data points: ['Africa', '0.4'], ['Arab States', '8.1'], ['Asia & Pacific', '14.4'], ['World', '14.9'], ['CIS', '19.8'], ['The Americas', '22.0'], ['Europe', '31.9']

Step 2: Select any 5 data points from the data.

The 5 data points are:

Data point 1: ['Africa', '0.4']

Data point 2: ['Europe', '31.9']

Data point 3: ['CIS', '19.8']

Data point 4: ['Asia & Pacific', '14.4']

Data point 5: ['World', '14.9']

Step 3: Generate questions about data points.

Question about Data point 1: What is the rate per 100 inhabitants in Africa?

Question about Data point 2: What is the rate per 100 inhabitants in Europe?

Question about Data point 3: Which region has rate per 100 inhabitants of 19.8?

Question about Data point 4: Which region has rate per 100 inhabitants of 14.4?

Question about Data point 5: What is the rate per 100 inhabitants in World?

Step 4: Final answer in the correct format:

1. What is the rate per 100 inhabitants in Africa?
2. What is the rate per 100 inhabitants in Europe?
3. Which region has rate per 100 inhabitants of 19.8?
4. Which region has rate per 100 inhabitants of 14.4?
5. What is the rate per 100 inhabitants in World?

"""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    # llm = HuggingFacePipeline(pipeline=llama_pipeline, verbose = True)
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
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
            Think step by step:
            """
            prompt_template = get_prompt(instruction)

            chain = PromptTemplate.from_template(prompt_template) | llm
            question = "Generate 5 short questions that can be answered only from this data: {chart_data}."
            question = question.format(chart_data=chart_data)
              
            print(chain.invoke({"question": question}))
            print("generating result from llm")

