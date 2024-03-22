import os
import csv
import torch
import transformers
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch
from langchain.document_loaders.csv_loader import CSVLoader
from torch import cuda

set_seed(42)

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
        top_k=5,
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

def setup_data_loader():
    loader = CSVLoader(file_path="/exports/eddie/scratch/s2024596/ragagent/dataset/data/5400.csv")
    return loader.load()

def setup_llm_chain(llama_pipeline):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest narrative writing assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations. Write a fluent narrative as a single paragraph with no subsection headings.
    """

    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template

    instruction = """
    Write a narrative based on a {chart_type} showing the following data: {chart_data} on the topic "{chart_title}".
    """

    template = get_prompt(instruction)

    prompt_template = PromptTemplate(template=template, input_variables=["chart_type", "chart_data", "chart_title"])

    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    return llm_chain

def process_data_and_run(llm_chain, chart_type, chart_data, chart_title):
    # model called in with the params here
    response = llm_chain({"chart_type": chart_type, "chart_data": chart_data, "chart_title": chart_title})
    print(response, "\n")
    return response

if __name__ == "__main__":
    # read in from new_xxx.py files

    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    print("llm setup")

    llm_chain = setup_llm_chain(llama_pipeline)
    print("LLM chain setup")

    sys.stdout.flush()

    chart_data_directory = "/home/s2024596/ragagent/dataset"
    with open("/home/s2024596/ragagent/dataset/new_output.txt", "r") as output_file:
        lines = output_file.read().splitlines()

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
            
            # # no need for description rn can be done seprately during evalutaion
            # with open('/home/s2024596/ragagent/dataset/chartdescription.txt', 'r') as file:
            #     chart_description_lines = file.readlines()
            #     chart_description = chart_description_lines[i].strip()

            response = process_data_and_run(llm_chain, chart_type, chart_data, chart_title)
            
            output_file_path = "/home/s2024596/ragagent/dataset/fluentexp1results.txt"

            with open(output_file_path, "a") as output_file:
                output_file.write(f"Index: {i}\n")
                output_file.write(f"Generated: {response}\n\n")
                output_file.flush()
