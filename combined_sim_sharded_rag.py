import torch
import os
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
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

set_seed(42)

print("entered the file")

def setup_huggingface():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_auth = "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"
    login(token=hf_auth)

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id, use_auth_token=hf_auth, config=model_config
    )

    return pretrained_model, tokenizer

# removed return full text = true
def setup_pipeline(pretrained_model, tokenizer):
    llama_pipeline = pipeline(
        task="text-generation",
        model=pretrained_model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.01,
        max_new_tokens=512,
        repetition_penalty=1.1,
    )

    return llama_pipeline


def setup_embeddings():
    embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        multi_process=True,
    )
    return hf


def setup_data_loader(path):
    loader = CSVLoader(
        file_path=path
        # file_path="/exports/eddie/scratch/s2024596/ragagent/dataset/data/5400.csv"
    )
    return loader.load()


def setup_vector_db(data_csv, hf):
    vectordb = Chroma.from_documents(documents=data_csv, embedding=hf)
    return vectordb


def setup_sharded_model(pretrained_model):
    accelerator = Accelerator()
    save_directory = "/exports/eddie/scratch/s2024596/ragagent/sharded_model"
    accelerator.save_model(
        model=pretrained_model, save_directory=save_directory, max_shard_size="2GB"
    )

    device_map = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    sharded_model = load_checkpoint_and_dispatch(
        pretrained_model,
        checkpoint=save_directory,
        device_map="auto",
        no_split_module_classes=["Block"],
    )

    return sharded_model


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.

An example of how to answer a question is shown below delimited by triple backticks. 
```
[INST]<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
<</SYS>>
Use the following pieces of context to answer the question at the end.
Year: 2012
Spending in billion U.S. dollars: 4.97

Year: 2017
Spending in billion U.S. dollars: 5.75

Year: 2013
Spending in billion U.S. dollars: 5.12

Question: By what percentage did spending decrease from 2012 to 2017?
Think step-by-step before answering.
Return the final answer as a sentence.
[/INST]

Step 1: Verify if the question is correct according to the data.

Spending in 2012 = $4.97 billion
Spending in 2017 = $5.75 billion
This shows the spending has increased from 2012 to 2017. Therefore, the question is incorrect. Calculate the percentage increase instead.

Step 2: Find the difference between the two values (spending in 2012 - spending in 2017).

Spending in 2012 = $4.97 billion
Spending in 2017 = $5.75 billion
So, the difference between the two values is:
$5.75 billion - $4.97 billion = $0.78 billion

Step 3: Divide the difference by the original value (spending in 2012) to get the percentage change.

Percentage change = ($0.78 billion / $4.97 billion) x 100%
Percentage change = 15.6%
Therefore, spending increased by 15.6 percent from 2012 to 2017.

Step 4: Return the answer.

Final answer: Spending increased by 15.6 percent from 2012 to 2017.
```

"""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def setup_prompt_template(instruction):
    return PromptTemplate.from_template(instruction)

def extract_questions(file_path):
    with open(file_path, "r") as file:
        questions_text = file.read()
    # Splitting the text into individual questions
    questions = questions_text.strip().split("\n\n")
    return questions

if __name__ == "__main__":
    pretrained_model, tokenizer = setup_huggingface()
    sharded_model = setup_sharded_model(pretrained_model)
    llama_pipeline = setup_pipeline(sharded_model, tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline)
    print("llm setup")
    hf = setup_embeddings()

    
    chart_data_directory = "/home/s2024596/ragagent/dataset"

    with open("/home/s2024596/ragagent/dataset/new_output.txt", "r") as output_file:
        lines = output_file.read().splitlines()[:2]

        for i, line in enumerate(lines):
            split = line.split("|")
            chart_type = split[0].strip()
            chart_data_filename = split[1].strip().replace('.txt', '.csv')

            if chart_type == "Simple":
                chart_data_path = os.path.join(chart_data_directory, "data", chart_data_filename)
            else: # if it is "Complex"
                chart_data_path = os.path.join(chart_data_directory, "multiColumn/data", chart_data_filename)
            
            data_csv = setup_data_loader(chart_data_path) # e.g. /exports/eddie/scratch/s2024596/ragagent/dataset/data/5400.csv
            
            vectordb = setup_vector_db(data_csv, hf)
            print("vector db collection count: ", vectordb._collection.count())
            print("chroma done")

            # Build prompt
            instruction = """
            Use the following pieces of context to answer the question at the end.
            {context}
            Question: {question}
            Think step-by-step before answering.
            Return the final answer as a sentence.
            """
            prompt_template = get_prompt(instruction)

            # Run RetrievalQA chain
            # removed chain_type_kwargs={
                    # "prompt": PromptTemplate.from_template(prompt_template),
                    # "verbose": True,
                # },
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate.from_template(prompt_template)
                },
            )
            print("retrieval qa built")
            
            file_path_for_qs = "/home/s2024596/ragagent/processed_just_improved_questions.txt"
            extracted_qs = extract_questions(file_path_for_qs)
            questions = extracted_qs[i].split("\n")
            
            overall_array = []
            for s in questions:
                overall_array.append(s[3:])
        
            for question in overall_array:
                result = qa_chain({"query": json.dumps(question)})
                print("generating result from retrieval qa for question:", question)
                print(result["result"])

            print("\n")

    # questions = [
    #     "By what percentage did spending decrease from 2012 to 2017?",
    #     "Has there been a consistent increase or decrease in spending from 2011 to 2017?",
    #     "What was the total spending in billion U.S. dollars in 2015?",
    #     "In which year was the highest spending recorded in billion U.S. dollars?",
    #     "What was the difference in spending between 2016 and 2014 in billion U.S. dollars?",
    # ]
