import transformers
import torch
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
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return llama_pipeline

def setup_embeddings():
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def setup_prompt_template():
    instruction = """
    Use the following pieces of context to answer the question at the end. Use your real-world information to explain the reason behind the trends in data.
    {context}
    Question: {question}
    Output data-driven narrative:
    """
    return PromptTemplate.from_template(instruction)

if __name__ == '__main__':
    pretrained_model, tokenizer = setup_huggingface()
    llama_pipeline = setup_pipeline(pretrained_model, tokenizer)
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
    Output data-driven narrative:
    """
    prompt_template = get_prompt(instruction)

    # Run RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
    )
    print("retrieval qa built")

    question = "Write a narrative based on a line chart showing the data passed in on the topic: Global spending on motorsports sponsorships 2011 to 2017 backed up by real world events that the data passed in suggests."
    result = qa_chain({"query": question})
    print("generating result from retrieval qa")
    print(result["result"])

