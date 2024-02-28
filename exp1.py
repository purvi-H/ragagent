print("reading the file")

import os
import sys
import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import PromptTemplate, LLMChain

model_id = "meta-llama/Llama-2-7b-chat-hf" 

hf_auth = "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"
login(token = hf_auth)

print("logged into hf")

model_config = transformers.AutoConfig.from_pretrained(
    model_id
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

print("tokenizer done")

pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    config=model_config)

print("model setup")

llama_pipeline = pipeline(
    task = "text-generation",  # LLM task
    model=pretrained_model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer = tokenizer,
    return_full_text = True, # langchain expects the full text
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

print("pipeline setup")

llm = HuggingFacePipeline(pipeline = llama_pipeline)


## Default LLaMA-2 prompt style

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

instruction = """
Write a narrative based on a {chart_type} showing the following data: {chart_data} on the topic "{chart_title}".
"""

template = get_prompt(instruction, sys_prompt)

chart_type = ""
chart_data = ""
chart_title = ""

prompt_template = PromptTemplate(template=template, input_variables=["chart_type", "chart_data", "chart_title"])

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

print("llm chain setup")

response = llm_chain.run(chart_type, chart_data, chart_title)

print(response)
