import os
import csv
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def setup_model(model_id, hf_auth):
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=hf_auth,
        config=model_config
    )
    return pretrained_model, tokenizer

def setup_llama_pipeline(pretrained_model, tokenizer):
    llama_pipeline = pipeline(
        task="text-generation",
        model=pretrained_model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer,
        return_full_text=True,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    return llama_pipeline

def setup_llm_chain(llama_pipeline):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
    Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations."""

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
    response = llm_chain({"chart_type": chart_type, "chart_data": chart_data, "chart_title": chart_title})
    print(response, "\n")

if __name__ == "__main__":
    print("Reading the file")

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_auth = "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"
    login(token=hf_auth)

    print("Logged into Hugging Face")

    pretrained_model, tokenizer = setup_model(model_id, hf_auth)
    print("Model setup")

    llama_pipeline = setup_llama_pipeline(pretrained_model, tokenizer)
    print("Pipeline setup")

    llm_chain = setup_llm_chain(llama_pipeline)
    print("LLM chain setup")

    chart_data_directory = "/home/s2024596/ragagent/dataset"
    with open("/home/s2024596/ragagent/dataset/output.txt", "r") as output_file:
        lines = output_file.read().splitlines()[:2]

        for i, line in enumerate(lines):
            split = line.split("|")
            chart_type = split[0].strip()
            chart_data_filename = split[1].strip().replace('.txt', '.csv')

            if chart_type == "Simple":
                chart_data_path = os.path.join(chart_data_directory, "data", chart_data_filename)
            else:
                chart_data_path = os.path.join(chart_data_directory, "multiColumn/data", chart_data_filename)

            with open(chart_data_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                chart_data = list(reader)

            with open('/home/s2024596/ragagent/dataset/charttitle.txt', 'r') as file:
                chart_title_lines = file.readlines()
                chart_title = chart_title_lines[i].strip()

            with open('/home/s2024596/ragagent/dataset/charttype.txt', 'r') as file:
                chart_type_lines = file.readlines()
                chart_type = chart_type_lines[i].strip()

            process_data_and_run(llm_chain, chart_type, chart_data, chart_title)
