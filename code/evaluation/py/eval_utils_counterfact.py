"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity

CUDA = "cuda:2"

def compute_rewrite_quality_counterfact_rs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """
    
    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    if type(target_new)!=dict:
        target_new = {"str":target_new}
    
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    # attribute_prompts = record["attribute_prompts"]
    # generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        # attribute_prompts
    ]
    # Flatten all the evaluated prefixes into one list.
    probs = test_batch_prediction(
        model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"]
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                # "attribute_prompts",
            ]
        )
    }
    
    #relation specific probs
    if "relation_specific_prompts" in record.keys():
        ret['relation_specific_prompts_probs'] = rs_metric_diff(model, tok, record)
    
    generation_test_interval = 10
    if snips is not None and record["case_id"] % generation_test_interval == 0:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret

def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

def rs_metric_diff(model, tok, record):
    is_llama = True if "llama" in model.config._name_or_path else False
    is_Llama = True if "Llama" in model.config._name_or_path else False
    
    relation_specific_prompts = record["relation_specific_prompts"] # l
    prefix_lens = [len(n) for n in tok(relation_specific_prompts)["input_ids"]] # l
    initial_tok_ids = record["rs_best"]
    initial_token = [tok.decode(i).strip(" ") for i in initial_tok_ids] # l
    target = record["requested_rewrite"]["target_new"]["str"] if type(record["requested_rewrite"]["target_new"])==dict else record["requested_rewrite"]["target_new"]
    target_token = [target] * len(initial_token)
    prompt_tok = tok(
        [
            f"{relation_specific_prompts[p]} {suffix}"
            for p in range(len(relation_specific_prompts))
            for suffix in [initial_token[p], target_token[p]]
        ],
        padding=True,
        return_tensors="pt",
    ).to(CUDA) # (l * 2)
    #TODO#
    if is_llama:
        t_token = tok(f"{target}")["input_ids"][1:] # (l * 2)
    elif is_Llama:
        t_token = tok(f" {target}")["input_ids"][1:] # (l * 2)
    else:
        t_token = tok(f" {target}")["input_ids"]
    target_len = len(t_token) # (l * 2)
    
    with torch.no_grad():
        logits = model(**prompt_tok).logits # (l * 2) * d1 * d2

    results = np.zeros((logits.size(0),), dtype=np.float32) #l * 2
    for i in range(logits.size(0)):  #(l * 2)
        cur_len = 1 if i % 2==0 else target_len
        for j in range(cur_len):
            cur_tok = initial_tok_ids[i // 2] if i % 2 == 0 else t_token[j]
            r_tmp = -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
            results[i] = max(results[i], r_tmp)
    return [
            {"target_true": results[i].item(), "target_new": results[i + 1].item()}
            for i in range(0, len(results), 2)
        ]

def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """ """
    is_llama = True if "llama" in model.config._name_or_path else False
    is_Llama = True if "Llama" in model.config._name_or_path else False

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(CUDA)

    if is_llama:
        a_tok, b_tok = (tok(f"{n}")["input_ids"][1:] for n in [target_new, target_true])
    elif is_Llama:
        a_tok, b_tok = (tok(f" {n}")["input_ids"][1:] for n in [target_new, target_true])
    else:
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
