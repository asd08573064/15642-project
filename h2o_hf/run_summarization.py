import argparse
import json
import os.path

import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses
# from xopen import xopen
import math
import matplotlib.pyplot as plt 

from rouge import Rouge
import logging
import numpy as np

# from lost_in_the_middle.prompting import (
#     Document,
#     get_closedbook_qa_prompt,
#     get_qa_prompt,
#     get_qa_prompt_index
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from transformers.models.llama.configuration_llama import LlamaConfig
from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention, LSHLlamaForCausalLM_streaming, LSHLlamaAttention_streaming



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": None,
    "llama_h2o": H2OLlamaForCausalLM
}

TAGET_MODULE = {
    "llama": None,
    "llama_h2o": H2OLlamaAttention
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--hh_size", type=int, default=1024)
    parser.add_argument("--recent_size", type=int, default=1024)

    parser.add_argument("--inference_type", type=str, default="h2o")
    ## LSH KV-Cache
    parser.add_argument("--num_buckets", type=int, default=64)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--num_hashes", type=int, default=32)
    parser.add_argument("--most_recent_seq_len", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--k_lsh", type=int, default=16)
    parser.add_argument("--k", type=int, default=0)

    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.batch_size>1:
        tokenizer.pad_token = tokenizer.eos_token

    if args.inference_type == "h2o":
        print('Enabling H2O KV cache')
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_h2o'].from_pretrained(model_name, config=config, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    elif args.inference_type == "lsh":
        print('Enabling LSH KV cache')
        config.batch_size = 1
        config.n_buckets = args.num_buckets
        config.threshold = args.threshold
        config.n_hashes = args.num_hashes
        config.most_recent_seq_len = args.most_recent_seq_len
        config.num_heads = args.num_heads
        config.k = args.k_lsh
        print("Loading model with LSH KV cache")
        print("num_buckets: {}, threshold: {}, num_hashes: {}, most_recent_seq_len: {}, num_heads: {}, k_lsh: {}".format(
            args.num_buckets, args.threshold, args.num_hashes, args.most_recent_seq_len, args.num_heads, args.k_lsh))
        model = LSHLlamaForCausalLM_streaming.from_pretrained(
            args.model_name,
            device_map="auto",
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Model loaded")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['article']
            label = request['summary_gt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_k=args.k,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
            )

            if  args.inference_type == "h2o":   
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE['llama_h2o']):
                        m._clean_cache()
                        
            elif args.inference_type == "lsh":
                for name, m in model.named_modules():
                    if isinstance(m, LSHLlamaAttention_streaming):
                        m._clean_cache()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]
            
            print('Prompt: {}'.format(prompt))
            print('Label: {}'.format(label))
            print('Generated: {}'.format(generate_text))

            scores = rouge.get_scores(generate_text, label)[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])

            
            drop_rate_stats = {}
            
            result['result'] = {
                "metrics": {
                    "rouge-1": scores['rouge-1']['f'],
                    "rouge-2": scores['rouge-2']['f'],
                    "rouge-l": scores['rouge-l']['f']
                },
                "choices": [
                    {
                        "text": generate_text,
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1
                },
                "drop_rate_stats": drop_rate_stats
            }
            
            if args.inference_type == "lsh":
                for name, m in model.named_modules():
                    if isinstance(m, LSHLlamaAttention_streaming):
                        # print out drop rate for each layer
                        stats = m.get_drop_rate_stats()
                        drop_rate_stats[name] = stats
                        print('Layer {} Drop Rate: {}'.format(name, stats))
                        m._clear_drop_rate_meter()
                result['result']['drop_rate_stats'] = drop_rate_stats
                        
          
            
            print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
                        
            results.append(result)
            

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    with open(output_path + '.rouge', 'w') as f:
        f.write('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))