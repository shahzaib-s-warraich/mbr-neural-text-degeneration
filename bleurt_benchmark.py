from transformers import pipeline, GenerationConfig, GPT2Tokenizer, GPT2LMHeadModel
from evaluate import load

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

# Load metric
metric = load('bleurt', module_type="metric")

# Configure sampling configuration
generation_config_eta = GenerationConfig(do_sample=True, eta_cutoff=0.9, max_new_tokens=50)
generation_config_eps = GenerationConfig(do_sample=True, epsilon_cutoff=0.9, max_new_tokens=50)
generation_config_top_p = GenerationConfig(do_sample=True, top_p=0.1, max_new_tokens=50)
generation_config_top_k = GenerationConfig(do_sample=True, top_k=3, max_new_tokens=50)

sampling_configs = {"eta": generation_config_eta, "eps": generation_config_eps, "top-p": generation_config_top_p, "top-k": generation_config_top_k}

# Set up text generator
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Inference
for name, sampling_config in sampling_configs.items():
    output = generator("Joe Biden", generation_config=sampling_config)[0]['generated_text']
    print(f"response:", output)
    res = metric.compute(predictions=[output], references=["Joseph Robinette Biden Jr. is an American politician who is the 46th and current president of the United States. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama and represented Delaware in the United States Senate from 1973 to 2009."])
    print(f"bluert benchmark for {name} sampling:", res)