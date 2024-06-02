import pandas as pd
from typing import *
import numpy as np
from evaluate import load
from tqdm import tqdm
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def transform_inputfile_to_text_sentences(filename: str) -> List[str]:
    """
    This function takes a filename as input and returns a list of sentences
    :param filename: the name of the file
    :return: a list of sentences
    """
    df = pd.read_json(filename, lines=True)
    gens = df["model_generations"]
    total_list = []

    for genl in gens:
        total_list += genl
    return total_list


def conditional_perplexity(generations_df, model, tokenizer, device="cuda"):
    # Modified version of https://github.com/alisawuffles/DExperts/blob/main/scripts/evaluation/evaluate_generations.py
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc="Evaluating fluency",
    ):
        prompt = row["prompt"]["text"]
        # print(prompt)
        prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # the multiplication is to get the total (we will use that to subtract later)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids).loss * (
            prompt_input_ids.shape[1] - 1
        )

        # for every generation conditioned on the prompt
        generations = [g["text"] for g in row["generations"]]
        for gen in generations:
            # print(gen)
            # print(prompt_input_ids.shape)
            full_input_ids = tokenizer.encode(gen, return_tensors="pt").to(device)
            full_loss = model(full_input_ids, labels=full_input_ids).loss * (
                full_input_ids.shape[1] - 1
            )

            # conditional loss
            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )
            # print(full_loss.item(), prompt_loss.item())
            # print(loss.item(), full_input_ids.shape[1], prompt_input_ids.shape[1])
            ppl = math.exp(loss.item())
            # print(ppl)
            if ppl < 1e4:  # for sanity
                perplexities.append(ppl)
            # else:
            # print("FUCK FUCK FUCK")
    # return mean and variance
    return np.nanmean(perplexities), np.nanvar(perplexities)


# def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
#     perplexities = []
#     ct = 0
#     # for every prompt
#     for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating fluency'):
#         prompt = row.prompt['text']
#         prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
#         prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
#         # for every generation conditioned on the prompt
#         generations = [g['text'] for g in row['generations']]
#         for gen in generations:
#             full_input_ids = tokenizer.encode(gen, return_tensors='pt').to(device)
#             full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
#             loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
#             ppl = math.exp(loss.item())
#             if ppl < 1e4:   # for sanity
#                 perplexities.append(ppl)
#     return np.nanmean(perplexities)


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc="Evaluating diversity",
    ):
        generations = [
            g["text"].replace(row["prompt"]["text"], "") for g in row["generations"]
        ]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(" ")
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + "_" + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + "_" + o[i + 1] + "_" + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def calc_perplexity(
    sequences: List[str], model_id: str, device: str, batch_size: int = 16
) -> float:
    """
    This function takes a list of sentences and a ngram model and returns the perplexity
    :param sequences: a list of sentences
    :param hugfacemodelname: the name of a causal model in huggingface
    """
    perplexity_calculater = load("perplexity", module_type="measurement")
    results = perplexity_calculater.compute(
        data=sequences, model_id=model_id, batch_size=batch_size, device=device
    )
    results_np_arr = np.array(results["perplexities"])
    return np.mean(results_np_arr)


def main(
    filename: str,
    model_id: str,
    device: str,
    batch_size: int = 16,
    num_entries: int = -1,
    calculate_perplexity: bool = False,
    calculate_diversity: bool = False,
    outdir: str = ".",
):
    """
    This function takes a filename as input and returns the perplexity
    :param filename: the name of the file
    :return: the perplexity
    """
    # sequences = transform_inputfile_to_text_sentences(filename)
    # perplexity = calc_perplexity(
    #     sequences=sequences, model_id=model_id, device=device, batch_size=batch_size
    # )
    # print(perplexity)
    output_dict = {"filename": filename}
    df = pd.read_json(filename, lines=True)

    if num_entries > 0:
        df_perp = df.sample(num_entries, random_state=42)

    if calculate_perplexity:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        perp = 0.0
        with torch.no_grad():
            perp, var_perp = conditional_perplexity(df_perp, model, tokenizer, device)

        output_dict["fluency"] = perp
        output_dict["var_fluency"] = var_perp

    if calculate_diversity:
        dist1, dist2, dist3 = distinctness(df)
        output_dict["dist1"] = dist1
        output_dict["dist2"] = dist2
        output_dict["dist3"] = dist3

    from pprint import pprint

    pprint(output_dict)

    import os

    out_file = os.path.join(outdir, os.path.basename(filename))

    import json

    with open(out_file, "w") as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=False)
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_entries", type=int, default=-1)
    parser.add_argument("--calculate_perplexity", action="store_true")
    parser.add_argument("--calculate_diversity", action="store_true")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()
    main(
        filename=args.filename,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        num_entries=args.num_entries,
        calculate_perplexity=args.calculate_perplexity,
        calculate_diversity=args.calculate_diversity,
        outdir=args.outdir,
    )
