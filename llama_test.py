from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.greedy_search import greedy_search,load_stop_words
from utils.cad import context_aware_decoding
from utils.load_prompt import load_prompt
from utils.load_save_path import load_save_path
from copy import deepcopy
from peft import PeftModel
import os
import json
import random
import torch
random.seed(42)
from tqdm import tqdm

import logging
# logging with filename line number time logging level and message
logging.basicConfig(level=logging.INFO,format='%(filename)s[line:%(lineno)d] %(asctime)s %(levelname)s %(message)s')

import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default = "/data3/MODELS/gpt-neox-20b")
parser.add_argument("--device", type=int, default=7)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str)
parser.add_argument("--example_file", type=str, required=True)
parser.add_argument("--example_number", type=int, default=1)
parser.add_argument("--prompt-name", type=str)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--decode_type", type=str, default="greedy", choices=["greedy","cad"])
parser.add_argument("--execute_delay_hour", type=float, default=None)
parser.add_argument("--debug", action="store_true",default=False)



args = parser.parse_args()
model_path = args.model
model_name = model_path.split("/")[-1]
lora_path = args.lora_path
if lora_path is not None:
    model_name = lora_path.split("/")[-1]
device = f"cuda:{args.device}"
example_number = args.example_number
decode_type = args.decode_type
execute_delay_hour = args.execute_delay_hour
input_file = args.input_file
example_file = args.example_file
prompt_name = args.prompt_name
prompt_name = prompt_name if prompt_name is not None else model_name

output_file = args.output_file
if output_file is None:
    output_file = f"results/{model_name}_{decode_type}_{prompt_name}_{example_number}.json"

dataset = json.load(open(input_file))
examples = json.load(open(example_file))

logging.info("model_path:",model_path)
logging.info("lora_path:",lora_path)
logging.info("device:",device)
logging.info("decode_type:",decode_type)
logging.info("input_file:",input_file)
logging.info("example_file:",example_file)
logging.info("prompt_name:",prompt_name)


if execute_delay_hour is not None:
    logging.info("sleeping for",execute_delay_hour,"hours")
    time.sleep(execute_delay_hour*3600)
    logging.info("wake up")


logging.info("example_number:",example_number)




if 'llama' in model_path or 'alpaca' in model_path:
    if '65b' in model_path or '30b' in model_path or '70b' in model_path or '34b' in model_path:
        # get all the cuda devices
        cuda_device_number = torch.cuda.device_count()
        logging.info("We have", cuda_device_number, "GPUs!")
        model = LLM(model=model_path,tensor_parallel_size=cuda_device_number,dtype='bfloat16')
        device = "cuda"
    else:
        model = LLM(model=model_path,dtype='bfloat16',gpu_memory_utilization=0.9)
        # model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

elif "20b" in model_path.lower():
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda:0"
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    
if lora_path is not None:
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        # torch_dtype=torch.float16,
    )

prompt_template = load_prompt(prompt_name,example_number)
logging.info("prompt_template:\n",prompt_template)

stop_words = load_stop_words(model_name)
stop_words.append("}}.")
logging.info("stop_words:",stop_words)

if args.debug:
    # set logging level to debug
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("debug mode, only use 10 examples")
    dataset = dataset[:10]

for unit in tqdm(dataset):
    
    passage = unit['passage']
    question = unit['question']
    prompt = deepcopy(prompt_template)
    
    unit_examples = random.sample(examples,example_number)
    for example_idx,random_example in enumerate(unit_examples,start=1):
        example_passage = random_example['passage']
        example_question = random_example['question']
        example_answer = random_example['rationale']
        prompt = prompt.replace(f"[[example_passage_{example_idx}]]", example_passage)
        prompt = prompt.replace(f"[[example_question_{example_idx}]]", example_question)
        prompt = prompt.replace(f"[[example_answer_{example_idx}]]", example_answer)
    
    prompt = prompt.replace("[[passage]]", passage)
    prompt = prompt.replace("[[question]]", question)
    logging.debug("="*100)
    logging.debug("prompt:\n",prompt)
    if decode_type == "greedy":
        sampling_params = SamplingParams(temperature=0, top_p=1.0,max_tokens=128,stop=stop_words)
        vllm_output = model.generate(prompt,sampling_params,use_tqdm=False)
        output = vllm_output[0].outputs[0].text
        logging.debug("vllm_output:\n",output)
    elif decode_type == "cad":
        question_input_ids = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        output = context_aware_decoding(input_ids,question_input_ids, model, tokenizer,max_length=128,stop_words=stop_words)
    unit[f'{model_name}_output'] = output
    
# store the results to different folder for different models
save_path = load_save_path(model_name,input_file,example_number,prompt_name,decode_type)

with open(save_path, "w") as f:
    json.dump(dataset, f, indent=4,ensure_ascii=False)
    