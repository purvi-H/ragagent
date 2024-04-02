from evaluate import load

def generate_gold_and_generated():
    gold = []
    generated = []
    # Read the text file
    with open("/home/s2024596/ragagent/dataset/dataset/baseline_13b_processed_results.txt", "r") as file:
        lines = file.readlines()

    # Process the lines to extract gold and generated pairs
    i = 0
    while i < len(lines):
        if lines[i].startswith("Gold:"):
            gold_text = lines[i][5:].strip()
            generated_text = lines[i+1][11:].strip()
            gold.append(gold_text)
            generated.append(generated_text)
            i += 2  # Move to the next pair
        else:
            i += 1  # Move to the next line
    return gold, generated

def calc_perplexity(generated): #generated = list of list
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=generated, model_id = "meta-llama/Llama-2-7b-chat-hf")
    return results

if __name__ == "__main__":
    gold, generated = generate_gold_and_generated()
    results = calc_perplexity(generated)
    print(results["perplexities"])
    print("\n")
    print(round(results["mean_perplexity"], 3))
