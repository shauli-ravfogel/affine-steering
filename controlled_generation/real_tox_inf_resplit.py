"""
The purpose of this file is to add code that runs inference real-toxocity-dataset splits to match the evaluations conducted in that paper.
"""

from datasets import load_dataset
import torch
from typing import *
from concept_erasure import LeaceFitter
from scripts_infra.dispatch import initialize_model_tok
import argparse
import pandas as pd
from scripts_infra.perspective import process_text_perspective, initialize_client
from googleapiclient.errors import HttpError
from scripts_infra.meta_tox_script import add_useful_columns, get_tox_numbers_from_df
import os
import json
from time import time


def process_sentence(
    text: str,
    num_generations: int,
    tokenizer,
    model,
    pipe,
    max_length: int,
    mode: Optional[str] = None,
) -> dict:
    """
    Takes in a sentence and turns it into a prompt for the models
    and then generates a bunch of completions. Returns a dictionary
    - text: the original text
    - prompt: the prompt used
    - completions: the completions generated
    """

    prompt = text
    prompt_len = len(tokenizer.encode(prompt))
    completions = None
    completions = pipe(
        prompt,
        num_return_sequences=num_generations,
        max_length=max_length + prompt_len,
        do_sample=True,
        top_p=0.9,
        top_k=0,
    )

    completion_list = [{"text": comp["generated_text"]} for comp in completions]

    return {"generations": completion_list}


def process_row(
    row,
    tokenizer,
    model,
    pipe,
    max_length: int,
    num_generations: int,
    mode: Optional[str] = None,
) -> dict:
    """
    Takes in a row from the dataset and processes it
    """
    print(type(row))
    text = row["prompt"]["text"]
    row_dict_cp = row.copy()

    processed = process_sentence(
        text=text,
        num_generations=num_generations,
        tokenizer=tokenizer,
        model=model,
        pipe=pipe,
        max_length=max_length,
        mode=mode,
    )

    # row_dict_cp.update(pd.Series(processed))
    # print(processed['completions'])
    # print(row_dict_cp['completions'])

    for key in processed:
        row_dict_cp[key] = processed[key]

    return row_dict_cp


def process_row_perspective_tup(args):
    return process_row_perspective(*args)


def process_row_perspective(row, api_key, client, full_sentence):
    # og_text = row["text"]
    prompt = row["prompt"]["text"]
    completions = row["generations"]

    # og_text_dict, prompt_dict
    try:
        prompt_perspective = process_text_perspective(prompt, api_key, client)
        prompt_dict = row["prompt"]
        prompt_dict.update(prompt_perspective)
        row["prompt"] = prompt_dict

    except HttpError as err:
        print(row)

    completions_dict_list = []
    for i, completion_dict in enumerate(completions):
        completion = completion_dict["text"]
        if not full_sentence:
            completion = completion.replace(prompt, "")
        try:
            completion_perspective = process_text_perspective(
                completion, api_key, client
            )
            completion_dict = {"text": completion}
            completion_dict.update(completion_perspective)
            completions_dict_list.append(completion_dict)
        except HttpError as err:
            continue
        except:
            continue

    row["generations"] = completions_dict_list

    return row


def summary_report_txt(
    pers_filename: str,
    model_id,
    dataset_file,
    num_generations,
    max_length,
    mode,
    intervention,
    partition: str,
    num_entries: int = -1,
):
    """
    Takes in a filename and prints out a summary report
    """
    df = pd.read_json(pers_filename, lines=True)
    print(df.head())
    print(df.columns)
    print(df.describe())
    print(df.info())

    df = add_useful_columns(df, column_name="generations")
    df = select_partition(df, partition)

    (
        tox_prompt_avg_max_tox_prompt,
        tox_prompt_var_max_tox_prompt,
        tox_prompt_avg_max_prof_prompt,
        tox_prompt_avg_max_seve_prompt,
        tox_prompt_avg_max_sex_prompt,
        tox_prompt_avg_max_iden_prompt,
        tox_prompt_avg_max_ins_prompt,
        tox_prompt_avg_max_flirt_prompt,
        tox_prompt_avg_avg_tox_prompt,
        tox_prompt_avg_avg_prof_prompt,
        tox_prompt_avg_avg_seve_prompt,
        tox_prompt_avg_avg_sex_prompt,
        tox_prompt_avg_avg_iden_per_prompt,
        tox_prompt_avg_avg_ins_per_prompt,
        tox_prompt_avg_avg_flirt_per_prompt,
        tox_prob,
    ) = get_tox_numbers_from_df(df)

    report_dict = {
        "model_id": model_id,
        "dataset_path": dataset_file,
        "partition": partition,
        "num_entries": num_entries,
        # "challenging": challenging,
        "num_generations": num_generations,
        "max_length": max_length,
        "mode": mode,
        # "embedding_threshold": embed
        # ding_threshold,
        # "embedding_coefficient": embedding_coefficient,
        "avg_max_tox_prompt": tox_prompt_avg_max_tox_prompt,
        "tox_prob": tox_prob,
        "var_max_tox_prompt": tox_prompt_var_max_tox_prompt,
        "avg_max_prof_prompt": tox_prompt_avg_max_prof_prompt,
        "avg_max_seve_prompt": tox_prompt_avg_max_seve_prompt,
        "avg_max_sex_prompt": tox_prompt_avg_max_sex_prompt,
        "avg_max_iden_prompt": tox_prompt_avg_max_iden_prompt,
        "avg_max_ins_prompt": tox_prompt_avg_max_ins_prompt,
        "avg_max_flirt_prompt": tox_prompt_avg_max_flirt_prompt,
        "avg_avg_tox_prompt": tox_prompt_avg_avg_tox_prompt,
        "avg_avg_prof_prompt": tox_prompt_avg_avg_prof_prompt,
        "avg_avg_seve_prompt": tox_prompt_avg_avg_seve_prompt,
        "avg_avg_sex_prompt": tox_prompt_avg_avg_sex_prompt,
        "avg_avg_iden_per_prompt": tox_prompt_avg_avg_iden_per_prompt,
        "avg_avg_ins_per_prompt": tox_prompt_avg_avg_ins_per_prompt,
        "avg_avg_flirt_per_prompt": tox_prompt_avg_avg_flirt_per_prompt,
    }

    try:
        report_dict.update(intervention.report_dict())
    except AttributeError:
        pass

    return report_dict


def select_partition(df, partition):
    if partition == "challenging":
        df = df[df["challenging"] == 1]
    elif partition == "non_challenging":
        df = df[df["challenging"] == 0]
    elif partition == "tox":
        df = df[df["prompt_tox"] == 1]
    elif partition == "non_tox":
        df = df[df["prompt_tox"] == 0]
    return df


def base_output_file_name(
    base_model: str,
    intervention,
    num_generations,
    mode,
    max_length,
    partition,
    num_entries,
    custom_dataset_path=None,
):
    model_name = base_model.split("/")[-1]
    intervention_name = str(intervention)
    file_name = f"real_tox_inference_{model_name}_{intervention_name}_{num_generations}_{max_length}_{mode}_{partition}_{num_entries}"
    if custom_dataset_path is not None:
        custom_dataset_base = custom_dataset_path.split("/")[-1].split(".")[0]
        file_name += f"_{custom_dataset_base}"
    return file_name


def calc_num_tokens_generated(row, tokenizer):
    prompt = row["prompt"]["text"]
    completions = row["generations"]
    prompt_len = len(tokenizer.encode(prompt))
    completion_lens = [
        len(tokenizer.encode(comp["text"])) - prompt_len for comp in completions
    ]
    return sum(completion_lens)


def calc_num_tokens_generated_dataset(df, tokenizer):
    num_tokens = 0
    for i, row in df.iterrows():
        num_tokens += calc_num_tokens_generated(row, tokenizer)
    return num_tokens


def main(
    base_model: str,
    num_generations: int,
    max_length: int,
    mode: Optional[str] = None,
    device: str = "cpu",
    partition: str = "all",  # one of challenging, non_challenging, all, tox, non_tox
    num_entries: int = -1,  # -1 means all
    full_sentence: bool = False,
    **kwargs,
):
    only_perspective = kwargs.get("only_perspective", False)
    dataset, processed_dataset, base_file_name = None, None, None
    if not only_perspective:
        # Load the dataset

        custom_dataset_path = kwargs.get("custom_dataset_path", None)

        if custom_dataset_path is None:
            dataset = pd.read_json(
                "real-toxicity-prompts/real_tox_prompt_tox_split.json", lines=True
            )
        else:
            dataset = pd.read_json(custom_dataset_path, lines=True)

        dataset = select_partition(dataset, partition)

        if num_entries != -1:
            dataset = dataset.sample(num_entries, random_state=42)
        print("before cuda call")
        torch.cuda.empty_cache()
        model, tokenizer, pipe = initialize_model_tok(
            base_model=base_model, mode=mode, device=device, **kwargs
        )

        print(model.device, pipe.device)

        start = time()
        with torch.no_grad():
            # Add generations
            processed_dataset = dataset.apply(
                process_row,
                axis=1,
                tokenizer=tokenizer,
                model=model,
                pipe=pipe,
                max_length=max_length,
                num_generations=num_generations,
                mode=mode,
            )
        end = time()

        model_name = base_model.split("/")[-1]
        intervention = None
        try:
            intervention = model.intervention
        except AttributeError:
            intervention = "original"

        base_file_name = base_output_file_name(
            base_model,
            intervention,
            num_generations,
            mode,
            max_length,
            partition,
            num_entries,
            custom_dataset_path=custom_dataset_path,
        )

        file_name = f"{base_file_name}.jsonl"
        only_inference_time_benchmark = kwargs.get(
            "only_inference_time_benchmark", False
        )
        if only_inference_time_benchmark:
            print(processed_dataset.head())
            num_tokens_generated = calc_num_tokens_generated_dataset(
                processed_dataset, tokenizer
            )
            print(f"Number of tokens generated: {num_tokens_generated}")
            print(f"Time taken: {end - start}")
            print(f"Seconds per token: {(end - start) / (num_tokens_generated)}")
            res_dict = {
                "num_tokens_generated": num_tokens_generated,
                "time_taken": end - start,
                "seconds_per_token": (end - start) / (num_tokens_generated),
            }
            with open(f"time_runs/{base_file_name}_time.json", "w") as f:
                json.dump(res_dict, f)
            return

        processed_dataset.to_json(file_name, orient="records", lines=True)

    else:
        file_name = kwargs.get("file_name", None)
        if file_name is None:
            raise ValueError("Must provide file_name if only_perspective is True")
        processed_dataset = pd.read_json(file_name, lines=True)
        base_file_name = file_name.split(".")[0]
    # Add perspective scores
    API_KEY, client = initialize_client()
    # processed_dataset = processed_dataset.apply(
    #     process_row_perspective, axis=1, api_key=API_KEY, client=client
    # )

    from multiprocessing import Pool

    tmp_list = list(processed_dataset.iterrows())
    tmp_list_with_api_client = [
        (x[1], API_KEY, client, full_sentence) for x in tmp_list
    ]
    with Pool(8) as p:
        processed_dataset = p.map(process_row_perspective_tup, tmp_list_with_api_client)
    processed_dataset = pd.DataFrame(processed_dataset)

    file_name_pers = f"{base_file_name}_{full_sentence}_perspective.jsonl"
    processed_dataset.to_json(file_name_pers, orient="records", lines=True)

    generated_report = summary_report_txt(
        pers_filename=file_name_pers,
        model_id=base_model,
        dataset_file="real-toxicity-full",
        num_generations=num_generations,
        max_length=max_length,
        mode=mode,
        intervention=intervention,
        partition=partition,
        num_entries=num_entries,
    )

    report_file_name = f"reports/{base_file_name}_{full_sentence}_report.json"
    json.dump(generated_report, open(report_file_name, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--control_coefficient", type=float, default=None)
    parser.add_argument("--control_threshold", type=float, default=None)
    parser.add_argument("--eraser_fitter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_entries", type=int, default=-1)
    parser.add_argument("--partition", type=str, default="all")
    parser.add_argument("--num_iterations", type=int, default=-1)
    parser.add_argument("--control_vector", type=str, default=None)
    parser.add_argument("--classifier", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--full_sentence", dest="full_sentence", action="store_true")
    parser.add_argument("--custom_dataset_path", type=str, default=None)
    parser.add_argument(
        "--only_inference_time_benchmark",
        dest="only_inference_time_benchmark",
        action="store_true",
    )
    parser.set_defaults(full_sentence=False)
    parser.set_defaults(only_inference_time_benchmark=False)

    parser.add_argument(
        "--only_perspective", dest="only_perspective", action="store_true"
    )
    parser.set_defaults(only_perspective=False)

    parser.add_argument("--file_name", type=str, default=None)
    args = parser.parse_args()

    print("AAAAAAAA")
    print(args.full_sentence)
    main(
        base_model=args.model_id,
        num_generations=args.num_generations,
        max_length=args.max_length,
        mode=args.mode,
        device=args.device,
        partition=args.partition,
        num_entries=args.num_entries,
        eraser_fitter_path=args.eraser_fitter_path,
        control_coefficient=args.control_coefficient,
        control_threshold=args.control_threshold,
        num_iterations=args.num_iterations,
        d_expl=args.d_expl,
        control_vector=args.control_vector,
        classifier=args.classifier,
        layer=args.layer,
        full_sentence=args.full_sentence,
        alpha=args.alpha,
        only_perspective=args.only_perspective,
        file_name=args.file_name,
        custom_dataset_path=args.custom_dataset_path,
        alpha_distribution=args.alpha_distribution,
        unit_vectors=args.unit_vectors,
        force_positive_coefficients=args.force_positive_coefficients,
        coefficients_softmax=args.coefficients_softmax,
        only_inference_time_benchmark=args.only_inference_time_benchmark,
        means_sampler=args.means_sampler,
        lam=args.lam,
        num_samples=args.num_samples,
    )
