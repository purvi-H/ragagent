from pprint import pprint
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def calculate_prompt_perplexity(output):
    total_log_prob = 0
    total_tokens = 0

    for sentence in output:
        for _, log_prob in sentence:
            total_log_prob += log_prob
            total_tokens += 1

    average_log_prob = total_log_prob / total_tokens
    average_log_prob_tensor = torch.tensor(average_log_prob)  # Convert to a Tensor
    prompt_perplexity = torch.exp(-average_log_prob_tensor)

    return prompt_perplexity.item()

def setup_model(model_id, hf_auth):
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=hf_auth,
        config=model_config
    )
    pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id
    return pretrained_model, tokenizer

if __name__ == "__main__":
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_auth = "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"
    login(token=hf_auth)

    input_texts_0 = ["You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, complex trends, pattern synthesis, exceptions, commonplace concepts. Also, include domain-specific insights, current events, social and political context, explanations."]
    # 79.678955078125
    input_texts_1 = ["You are a helpful, respectful and honest assistant. Answer safely and respectfully. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data instead describe statistics, extrema, outliers, correlations, point-wise comparisons, trends, pattern synthesis, exceptions, commonplace concepts."]
    # 86.09090423583984
    input_texts_2 = ["You are a helpful, respectful and honest assistant. Answer safely and respectfully. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data. Instead describe statistics, extrema, outlier, correlations and commonplace concepts."]
    # 82.80303955078125
    input_texts_3 = ["You are a helpful, respectful and honest narrative writer. Answer safely and respectfully. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data. Instead describe statistics, extrema, outlier, correlations and commonplace concepts."]
    # 82.71334838867188
    input_texts_4 = ["You are a helpful, respectful and honest narrative writer. Always answer safely and respectfully. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data. Instead describe statistics, extrema, outlier, correlations and commonplace concepts."]
    # 80.9637451171875
    input_texts_4 = ["You are a helpful, respectful and honest narrative writer. Always answer helpfully and respectfully. Write a narrative which describes the information in the chart data. Do not discuss what is missing in the data, instead describe statistics, extrema, outlier, correlations and any jargons."]
    # 75.60726928710938

    pretrained_model, tokenizer = setup_model(model_id, hf_auth)
    print("Model setup")

    batch = to_tokens_and_logprobs(pretrained_model, tokenizer, input_texts_4)
    pprint(batch)

    prompt_perplexity = calculate_prompt_perplexity(batch)
    print("Prompt Perplexity:", prompt_perplexity)