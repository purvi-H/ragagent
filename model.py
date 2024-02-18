print("reading the file")

import os
import sys

import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline

model_id = "meta-llama/Llama-2-7b-chat-hf" 

# os.environ["HUGGINGFACE_TOKEN"] = "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"
# huggingface-cli login --token $HUGGINGFACE_TOKEN

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

prompt = 'What is the difference between fission and fusion?\n'

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

'''generate_text = llama_pipeline(
	prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max                                                                                    
        max_new_tokens=512,  # mex number of tokens to generate in the output                                                                                          
        repetition_penalty=1.1  # without this output begins repeating                                                                                                 
)
'''

print("pipeline setup")

''' def get_llama_response(prompt):
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    print("Respose: ", sequences[0]['generated_text'])
'''

# prompt = 'What is the difference between fission and fusion?\n'
print("generating response")
# get_llama_response(prompt)
# llm = HuggingFacePipeline(pipeline=generate_text)
llm = HuggingFacePipeline(pipeline = llama_pipeline)
print(llm(prompt=prompt))
