import torch
import transformers
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
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. """

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline, verbose=True)
    print("llm setup")

    instruction = """ Here is an example of how to write a narrative based on the context. The example is delimited by triple exclamation marks.
    Example:!!!Write a narrative based on the context for a bar chart about the topic: Current year in various historical and world calendars 2020.

Context:
What is the name of the calendar corresponding to the year 1441 as of January 25, 2020?
Answer: Islamic
Which calendar year is the highest among the given calendars as of January 25, 2020?
Answer: Assyrian (6770)
Which calendar has the lowest year value among the ones listed as of January 25, 2020?
Answer: French Revolutionary (228)
What is the difference in years between the Gregorian and the Julian calendar as of January 25, 2020?
Answer: 247
Which calendar year is closest to the Gregorian calendar year as of January 25, 2020?
Answer: Hindu (1941)

short data-driven narrative:

In the diverse tapestry of historical and world calendars for the year 2020, each system paints a unique picture of time. As of January 25, 2020, the Assyrian calendar reigns supreme with its lofty year count of 6770, while the French Revolutionary calendar lingers at the bottom with a humble 228. The Gregorian and Julian calendars, separated by 247 years, showcase the evolution of timekeeping systems over centuries. Amidst this variation, the Hindu calendar resonates closely with the Gregorian, mirroring the year 1941. However, it's the Islamic calendar that stands out with its current year of 1441, rooted in lunar cycles and religious tradition. Each calendar offers a distinct perspective on the passage of time, weaving together a rich tapestry of human history and culture.!!!
    Based on the Example of how to perform the task, perform the following task: {question}"""
    prompt_template = get_prompt(instruction)

    chain = PromptTemplate.from_template(prompt_template) | llm
    question = """Write a narrative based on the context for a bar chart about the topic: Food retail sales growth in the United Kingdom (UK) 2014 to 2018.

Context:

What was the retail sales growth in 2017?
Answer: 1.1%
Did retail sales growth increase or decrease from 2016 to 2015?
Answer: Decrease
What was the retail sales growth in 2018?
Answer: 0.9%
Which year had the highest retail sales growth among the given years?
Answer: 2014
By what percentage did retail sales growth change from 2014 to 2016?
Answer: Retail sales growth decreased by approximately 30.77% from 2014 to 2016.

short data-driven narrative:"""
    print("generating result from llm")
    print(chain.invoke({"question": question}))

