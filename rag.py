print("reading the file")

import os
import sys

import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

print("pipeline setup")

# prompt = 'What is the difference between fission and fusion?\n'
llm = HuggingFacePipeline(pipeline = llama_pipeline)
#print(llm(prompt=prompt))
print("llm setup")

from langchain.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
from torch import cuda

loader = CSVLoader(file_path="/exports/eddie/scratch/s2024596/ragagent/dataset/data/5400.csv")

data_csv = loader.load() # contains a list of objects of type <class 'langchain_core.documents.base.Document'>

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

'''
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id
)
'''
'''
model_kwargs = {'device': 'gpu'}
encode_kwargs = {'normalize_embeddings': False}
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
hf = HuggingFaceEmbeddings(
    model_name= embed_model_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    multi_process = True
)
print("model embedded")
'''

# Check if CUDA (GPU) is available and set the device accordingly
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Define model and encode kwargs
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}

# Initialize HuggingFaceEmbeddings instance
hf = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    multi_process=True
)

print("model embedded")

from langchain_community.vectorstores import Chroma
'''
persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=data_csv,
    embedding=hf,
    persist_directory=persist_directory
)

#print( "vector db collection count: ",vectordb._collection.count())
#print("chroma done")
'''

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

if __name__ == '__main__':
    print("entered main")
    persist_directory = 'docs/chroma/'

    # Assume data_csv and hf are defined elsewhere
    vectordb = Chroma.from_documents(
        documents=data_csv,
        embedding=hf,
        # persist_directory=persist_directory
    )
    print("vector db collection count: ", vectordb._collection.count())
    print("chroma done")

    # Build prompt
    instruction = """
    Use the following pieces of context to answer the question at the end. Use your real-world information to explain the reason behind the trends in data.
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

    question = "Write a narrative based on a line chart showing the data passed in on the topic: Global spending on motorsports sponsorships 2011 to 2017"
    result = qa_chain({"query": question})
    print("generating result from retrieval qa")
    print(result["result"])


# if __name__ == '__main__':
#     print("entered main")
#     persist_directory = 'docs/chroma/'

#     vectordb = Chroma.from_documents(
#             documents=data_csv,
#             embedding=hf,
#             # persist_directory=persist_directory
#     )
#     print( "vector db collection count: ",vectordb._collection.count())
#     print("chroma done")

#     # Build prompt
#     template = """Use the following pieces of context to answer the question at the end. Use your real-world information to explain the reason behind the trends in data.
#     {context}
#     Question: {question}
#     Output data-driven narrative: """

#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
#     # Run chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=vectordb.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )

#     print("retrieval qa built")

#     question = "Write a narrative based on a line chart showing the data passed in on the topic: Indonesia - Gross domestic product (GDP) per capita in current prices from 1984 to 2024"
#     result = qa_chain({"query": question})
#     print("generating result from retrieval qa")
#     print(result["result"])
