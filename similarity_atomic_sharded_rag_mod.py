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

def setup_embeddings():
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        multi_process=True
    )
    return hf

def setup_data_loader():
    loader = CSVLoader(file_path="/exports/eddie/scratch/s2024596/ragagent/dataset/data/5400.csv")
    return loader.load()

def setup_vector_db(data_csv, hf):
    vectordb = Chroma.from_documents(
        documents=data_csv,
        embedding=hf
    )
    return vectordb

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

def setup_prompt_template(instruction):
    return PromptTemplate.from_template(instruction)
"""
if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    print("llm setup")
    hf = setup_embeddings()
    data_csv = setup_data_loader()

    vectordb = setup_vector_db(data_csv, hf)
    print("vector db collection count: ", vectordb._collection.count())
    print("chroma done")

    # Build prompt
    instruction = "Use the following pieces of context to answer the question at the end. {context} Question: {question} Brief answer:"
    prompt_template = get_prompt(instruction)

    # Run RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="mmr"),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template), "verbose": True}
    )
    print("retrieval qa built")

    question = "What was the highest spending year during the period covered (2011-2017)?"
    result = qa_chain({"query": question})
    print("generating result from retrieval qa")
    print(result["result"])
"""
if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    print("llm setup")
    hf = setup_embeddings()
    data_csv = setup_data_loader()

    vectordb = setup_vector_db(data_csv, hf)
    print("vector db collection count: ", vectordb._collection.count())
    print("chroma done")

    # Build prompt
    instruction = """
    Use the following pieces of context to answer the question at the end.
    {context}
    Question: {question}
    Let's think step-by-step before answering:
    """
    prompt_template = get_prompt(instruction)

    # Run RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template), "verbose": True}
    )
    print("retrieval qa built")

    questions = [
        "By what percentage did spending decrease from 2012 to 2017?",
        "Has there been a consistent increase or decrease in spending from 2011 to 2017?",
        "What was the total spending in billion U.S. dollars in 2015?",
        "In which year was the highest spending recorded in billion U.S. dollars?",
        "What was the difference in spending between 2016 and 2014 in billion U.S. dollars?"
    ]
    for question in questions:
        result = qa_chain({"query": question})
        print("generating result from retrieval qa for question:", question)
        print(result["result"])

