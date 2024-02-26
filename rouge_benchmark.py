import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline, GenerationConfig

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)

model = T5ForConditionalGeneration.from_pretrained(model_name)

summarization_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_rouge = evaluate.load("rouge")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test").select(range(100))

# Configure sampling configuration
generation_config_eta = GenerationConfig(do_sample=True, eta_cutoff=0.9)
generation_config_eps = GenerationConfig(do_sample=True, epsilon_cutoff=0.9)
generation_config_top_p = GenerationConfig(do_sample=True, top_p=0.1)
generation_config_top_k = GenerationConfig(do_sample=True, top_k=3)

sampling_configs = {"eta":generation_config_eta, "eps":generation_config_eps, "top-p":generation_config_top_p, "top-k":generation_config_top_k}
rouge_benchmark = {'eta':0.0, 'eps':0.0, 'top-p':0.0, 'top-k':0.0}

for sampling, generation_config in sampling_configs.items():
    summaries = []
    outputs = summarization_pipeline(
        dataset["article"],
        generation_config=generation_config,
        tokenizer=tokenizer,
        max_length = 50,
        truncation=False,
        progress_bar=True,
        batch_size=32
    )

    for output in outputs:
        summaries.append(output["summary_text"])
    rouge_benchmark[sampling] = evaluation_metric_rouge.compute(predictions=summaries, references=dataset["highlights"])
print(rouge_benchmark)