import evaluate
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline, GenerationConfig
from mbr import MBR, MBRConfig

from copy import deepcopy
from pathlib import Path
import jsonlines
from tqdm import tqdm


results_file = jsonlines.open(Path(__file__).parent / f"results_mbr.jsonl", "w")

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1096)

model = MBR(T5ForConditionalGeneration).from_pretrained(model_name)

summarization_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_rouge = evaluate.load("rouge")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test").select(range(10))

# MBR
mbr_config = MBRConfig()
mbr_config.num_samples = 30
mbr_config.num_references = 30
mbr_config.metric = "rouge"
mbr_config.metric_output_field = "rouge1"

base_generation_config = GenerationConfig.from_pretrained(model_name)
generation_configs = {}

# eta=0.9
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.eta_cutoff=0.9
generation_configs["greedy_eta"] = generation_config

# eps=0.9
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.epsilon_cutoff=0.9
generation_configs["greedy_eps"] = generation_config

# nucleus=0.1
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.top_p=0.1
generation_configs["greedy_top-p"] = generation_config

# top-k=3
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.top_k=3
generation_configs["greedy_top-k"] = generation_config

for method, generation_config in generation_configs.items():
    print(method, flush=True)
    summaries = []
    outputs = summarization_pipeline(
        dataset["article"],
        mbr_config=mbr_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        max_length = 50,
        truncation=False,
        progress_bar=True,
        batch_size=32
    )

    for output in tqdm(outputs):
        summaries.append(output["summary_text"])
    rouge_score = evaluation_metric_rouge.compute(predictions=summaries, references=dataset["highlights"])
    results_file.write({
        "method": method,
        "rouge": rouge_score,
        "summaries": summaries,
    })

results_file.close()
