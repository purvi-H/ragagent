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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. """

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


# foo = "meow"
# bar = "another meow"

# question = f"""Generate 5 questions that can be answered only from this data about {foo}:\n{bar}"""
# print(question)