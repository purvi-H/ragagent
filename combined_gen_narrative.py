import sys
import torch
import transformers
import re
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

def extract_query_output(file_path):
    output_strings = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_output = ""
        for line in lines:
            if line.startswith("{'query':"):
                if current_output:
                    output_strings.append(current_output.strip())
                current_output = line.strip()
            else:
                current_output += line.strip()
        if current_output:
            output_strings.append(current_output.strip())
    return output_strings

def remove_special_characters(text):
    # Define regex pattern to match special characters
    pattern = r'[^\w\s?]'  # This pattern keeps letters, digits, and whitespaces
    
    # Use regex to replace special characters with an empty string
    cleaned_text = [re.sub(pattern, '', line) for line in text]
    
    return cleaned_text

def clean_and_group_contexts(queries):

    processed_contexts = []
    for query_output in queries:
        # Find the index of the substring ', 'source_documents'
        index_source_documents = query_output.find(", 'source_documents'")
        modified_string = query_output[:index_source_documents]
        i = modified_string.find("'result': '[INST]<<SYS>>")
        length_to_add = len("""\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n\nAn example of how to answer a question is shown below delimited by triple backticks. \n```\n[INST]<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \n<</SYS>>\nUse the following pieces of context to answer the question at the end.\nYear: 2012\nSpending in billion U.S. dollars: 4.97\n\nYear: 2017\nSpending in billion U.S. dollars: 5.75\n\nYear: 2013\nSpending in billion U.S. dollars: 5.12\n\nQuestion: By what percentage did spending decrease from 2012 to 2017?\nThink step-by-step before answering.\nReturn the final answer as a sentence.\n[/INST]\n\nStep 1: Verify if the question is correct according to the data.\n\nSpending in 2012 = $4.97 billion\nSpending in 2017 = $5.75 billion\nThis shows the spending has increased from 2012 to 2017. Therefore, the question is incorrect. Calculate the percentage increase instead.\n\nStep 2: Find the difference between the two values (spending in 2012 - spending in 2017).\n\nSpending in 2012 = $4.97 billion\nSpending in 2017 = $5.75 billion\nSo, the difference between the two values is:\n$5.75 billion - $4.97 billion = $0.78 billion\n\nStep 3: Divide the difference by the original value (spending in 2012) to get the percentage change.\n\nPercentage change = ($0.78 billion / $4.97 billion) x 100%\nPercentage change = 15.6%\nTherefore, spending increased by 15.6 percent from 2012 to 2017.\n\nStep 4: Return the answer.\n\nFinal answer: Spending increased by 15.6 percent from 2012 to 2017.\n```\n\n\n<</SYS>>\n\n\nmorwordsaddedfordmorwordsaddedfordmorwordsaddedformorwordsaddedfdsaddedfdsaddedforddfun""")
        before = i
        after = i + length_to_add
        updated = modified_string[:before] + modified_string[after:]
        processed_contexts.append(updated)

    updated_processed_contexts = []
    for answer in processed_contexts:
        intermediate = []
        splitted = answer.split("\\n")
        splitted = remove_special_characters(splitted)
        splitted[0] = splitted[0].replace("Use the following pieces of context to answer the question at the end", "")
        question = splitted[0]
        answer = splitted[-5:]
        intermediate.append(question)
        intermediate.append(answer)
        updated_processed_contexts.append(str(intermediate))
    
    grouped_contexts = [str(updated_processed_contexts[i:i+5]) for i in range(0, len(updated_processed_contexts), 5)]

    return grouped_contexts

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
You are a helpful, respectful and honest narrative writing assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations. Write a fluent narrative as a single paragraph with no subsection headings."""

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

    file_path = "/home/s2024596/ragagent/allrag.txt"
    query_outputs = extract_query_output(file_path)
    query_outputs = query_outputs[1:]
    contexts = clean_and_group_contexts(query_outputs) # list of 400 contexts type of each is a string

    
    # 
    with open("/home/s2024596/ragagent/dataset/new_output.txt", "r") as output_file:
        lines = output_file.read().splitlines()[:2]
        
        for i, line in enumerate(lines):
            split = line.split("|")

            title = split[2].strip()

            with open('/home/s2024596/ragagent/dataset/new_charttype.txt', 'r') as file:
                chart_type_lines = file.readlines()
                chart_type = chart_type_lines[i].strip()
            
            context = contexts[i]
            example = """['{'query': '"What is the name of the calendar corresponding to the year 1441 as of January 25, 2020?"', ['The calendar corresponding to the year 1441 as of January 25, 2020, is the Islamic calendar.', '', 'Final answer: The calendar corresponding to the year 1441 as of January 25, 2020, is the Islamic calendar.']}', '{'query': '"Which calendar year is the highest among the given calendars as of January 25, 2020?"', ['The highest calendar year among the given calendars as of January 25, 2020, is the Assyrian calendar with the year 6770.', '', 'Final answer: The highest calendar year among the given calendars as of January 25, 2020, is the Assyrian calendar with the year 6770.']}', '{'query': '"Which calendar has the lowest year value among the ones listed as of January 25, 2020?"', ['The calendar with the lowest year value among the ones listed as of January 25, 2020, is the French Revolutionary calendar with the year 228.', '', 'Final answer: The calendar with the lowest year value among the ones listed as of January 25, 2020, is the French Revolutionary calendar with the year 228.']}', '{'query': '"What is the difference in years between the Gregorian and the Julian calendar as of January 25, 2020?"', ['The difference in years between the Gregorian and the Julian calendar as of January 25, 2020, is 247 years.', '', 'Final answer: The difference in years between the Gregorian and the Julian calendar as of January 25, 2020, is 247 years.']}', '{'query': '"Which calendar year is closest to the Gregorian calendar year as of January 25, 2020?"', ['The calendar year closest to the Gregorian calendar year as of January 25, 2020, is the Hindu calendar with the year 1941.', '', 'Final answer: The calendar year closest to the Gregorian calendar year as of January 25, 2020, is the Hindu calendar with the year 1941.']}]"""
            narrative = """In the diverse tapestry of historical and world calendars for the year 2020, each system paints a unique picture of time. As of January 25, 2020, the Assyrian calendar reigns supreme with its lofty year count of 6770, while the French Revolutionary calendar lingers at the bottom with a humble 228. The Gregorian and Julian calendars, separated by 247 years, showcase the evolution of timekeeping systems over centuries. Amidst this variation, the Hindu calendar resonates closely with the Gregorian, mirroring the year 1941. However, it's the Islamic calendar that stands out with its current year of 1441, rooted in lunar cycles and religious tradition. Each calendar offers a distinct perspective on the passage of time, weaving together a rich tapestry of human history and culture."""



 
            instruction = """Here is an example of how to write a narrative based on the context. The example is delimited by triple exclamation marks.
                            Example:
                            !!!
                            Write a narrative based on the context for a bar chart about the topic: Current year in various historical and world calendars 2020.
                            Context:
                            query: What is the name of the calendar corresponding to the year 1441 as of January 25, 2020?
                            answer: The calendar corresponding to the year 1441 as of January 25, 2020, is the Islamic calendar. Final answer: The calendar corresponding to the year 1441 as of January 25, 2020, is the Islamic calendar.
                            query: Which calendar year is the highest among the given calendars as of January 25, 2020?
                            answer: The highest calendar year among the given calendars as of January 25, 2020, is the Assyrian calendar with the year 6770. Final answer: The highest calendar year among the given calendars as of January 25, 2020, is the Assyrian calendar with the year 6770.
                            query: Which calendar has the lowest year value among the ones listed as of January 25, 2020?
                            answer: The calendar with the lowest year value among the ones listed as of January 25, 2020, is the French Revolutionary calendar with the year 228. Final answer: The calendar with the lowest year value among the ones listed as of January 25, 2020, is the French Revolutionary calendar with the year 228.
                            query: What is the difference in years between the Gregorian and the Julian calendar as of January 25, 2020?
                            answer: The difference in years between the Gregorian and the Julian calendar as of January 25, 2020, is 247 years. Final answer: The difference in years between the Gregorian and the Julian calendar as of January 25, 2020, is 247 years.
                            query: Which calendar year is closest to the Gregorian calendar year as of January 25, 2020?
                            answer: The calendar year closest to the Gregorian calendar year as of January 25, 2020, is the Hindu calendar with the year 1941. Final answer: The calendar year closest to the Gregorian calendar year as of January 25, 2020, is the Hindu calendar with the year 1941.
                            
                            Short fluent data-driven narrative: In the diverse tapestry of historical and world calendars for the year 2020, each system paints a unique picture of time. As of January 25, 2020, the Assyrian calendar reigns supreme with its lofty year count of 6770, while the French Revolutionary calendar lingers at the bottom with a humble 228. The Gregorian and Julian calendars, separated by 247 years, showcase the evolution of timekeeping systems over centuries. Amidst this variation, the Hindu calendar resonates closely with the Gregorian, mirroring the year 1941. However, it's the Islamic calendar that stands out with its current year of 1441, rooted in lunar cycles and religious tradition. Each calendar offers a distinct perspective on the passage of time, weaving together a rich tapestry of human history and culture.
                            !!!
                            
                            Based on the Example of how to perform the task, perform the following task: {question}"""
            # instruction = instruction.format(example=example, narrative=narrative)

            prompt_template = get_prompt(instruction)

            chain = PromptTemplate.from_template(prompt_template) | llm

            question = """Write a narrative based on the context for a {chart_type} about the topic: {title} 
            Context is answers to questions asked by user: {context}
            Short fluent data-driven narrative:: """

            question = question.format(chart_type=chart_type, title = title, context = context)
            
            print("generating result from llm")   
            print(chain.invoke({"question": question}))
            # response = chain({"question": question})
            print("\n")
            sys.stdout.flush()

