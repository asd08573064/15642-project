import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json

from utils_real_drop.modify_llama import H2OLlamaForCausalLM_streaming, LSHLlamaForCausalLM_streaming

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")
    
    ## LSH KV-Cache
    parser.add_argument("--num_buckets", type=int, default=64)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--num_hashes", type=int, default=32)
    parser.add_argument("--most_recent_seq_len", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    ## H2O KV-Cache
    parser.add_argument("--heavy_hitter_size", type=int, default=1)
    parser.add_argument("--enable_h2o_kv_cache", action="store_true")

    args = parser.parse_args()
    return args


def load(model_name_or_path, inference_type="", args=None):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if args is not None:
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.hh_size = args.heavy_hitter_size
        config.recent_size = args.recent_size

    if inference_type == "heavy_hitter":
        print("Loading H2O Llama model ...")
        model = H2OLlamaForCausalLM_streaming.from_pretrained(
        model_name_or_path,
        device_map="auto",
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    elif inference_type == "lsh":
        print("Loading LSH Llama model ...")
        config.batch_size = 1
        config.n_buckets = 64
        config.threshold = 2
        config.n_hashes = 32
        config.most_recent_seq_len = 128
        config.num_heads = 32
        config.k = 16
        model = LSHLlamaForCausalLM_streaming.from_pretrained(
        model_name_or_path,
        device_map="auto",
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict