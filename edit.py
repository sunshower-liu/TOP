import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
from util import nethook
from util.generate import generate_interactive, generate_fast
from demo import demo_model_editing, stop_execution, print_loud
import pdb
import numpy as np
import json
import random
from util.globals import *
import argparse

CUDA = "cuda:0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-name', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()
    
    # MODEL_NAME = "gpt2-xl", "gpt-j-6b", "Qwen2.5-14B", "Qwen2.5-7B-Instruct"
    MODEL_NAME = args.model
    model, tok = (
    AutoModelForCausalLM.from_pretrained(f"{MODEL_NAME}").to(CUDA),
    AutoTokenizer.from_pretrained(f"{MODEL_NAME}"),
    )
    
    config = model.config
    tok.pad_token = tok.eos_token
    
    if 'llama' in MODEL_NAME.lower():
        tok.padding_side = 'right'
    request = [
    {
        "prompt": "{}",
        "subject": "Lionel Messi is a citizen of",
        "target_new": {"str": "China"},
    }
    ]
    generation_prompts = [
        "What is the citizenship of Lionel Messi?",
        "Lionel Messi is a citizen of"
    ]
    
    # Execute rewrite
    ALG_NAME = args.alg_name
    
    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME, model_name=MODEL_NAME
    )
    
    model_new.config = config
    if args.model in ["gpt-j-6b", "gpt2-xl"]:
        generate_interactive(model_new, tok, max_out_len=200, use_logit_lens=True)
    else:
        generate_interactive(model_new, tok, max_out_len=200, use_logit_lens=True, ln_f_module="model.norm", layer_module_tmp="model.layers.{}", lm_head_module="lm_head")
    return

    
if __name__ == "__main__":
    main()