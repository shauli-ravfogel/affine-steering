from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers import pipeline, set_seed
import torch
from torch.utils.data import Dataset, DataLoader
from modeling_backpack_gpt2 import BackpackGPT2LMHeadModel
import argparse
from typing import *
from datasets import load_dataset
import json
import pandas as pd
from custom_gpt import CustomGPT2Model, CustomGPT2LMHeadModel
from concept_erasure import LeaceFitter
from try_control_cons_opt import ControlGeneration

set_seed(42)

MODEL_CHOICES = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
DEFAULT_CHOICE = MODEL_CHOICES[1]
BACKPACKS_MODEL_ID = "stanfordnlp/backpack-gpt2"
DEVICE = "cuda"
DEVICE_ID = 1
MAX_LENGTH = 512


def initialize_data(challenging_only: str = "challenging") -> Dataset:
    """
    Initializes the dataset for the baseline inference model.
    The Real Toxicity prompts dataset.
    :param challenging_only: filter to get only the challenging prompts.
    """

    real_toxic = load_dataset("allenai/real-toxicity-prompts")
    if challenging_only == "challenging":
        real_toxic_challanging = real_toxic.filter(lambda x: x["challenging"] == True)
        return real_toxic_challanging["train"]
    elif challenging_only == "non":
        real_toxic_non_challanging = real_toxic.filter(
            lambda x: x["challenging"] == False
        )
        return real_toxic_non_challanging["train"]
    elif challenging_only == "all":
        return real_toxic["train"]


def initialize_data_from_json(
    file_path: str,
    challenging_only: str = "challenging",
    non_challenging_ratio: Optional[float] = None,
) -> pd.DataFrame:
    """
    Initializes the dataset for the baseline inference model.
    The Real Toxicity prompts dataset.
    :param challenging_only: filter to get only the challenging prompts.
    """

    df = pd.read_json(file_path, lines=True)
    # if challenging_only:
    if challenging_only == "challenging":
        df = df.query("challenging == True")
        return df
    elif challenging_only == "non":
        df_challenging = df.query("challenging == True")
        df_non_challenging = df.query("challenging == False")
        non_challenging_number = int(len(df_challenging) * non_challenging_ratio)
        df_non_challenging = df_non_challenging.sample(
            n=non_challenging_number, random_state=42
        )
        return df_non_challenging
    elif challenging_only == "all":
        return df


def process_prompt(
    prompt: str,
    num_generations: int,
    tokenizer,
    model,
    pipe,
    max_length: int,
    control_vector=None,
    control_coefficient=0.0,
    control_threshold=None,
    mode=None,
    eraser=None,
    proj=None,
    Ws=None,
    tox_classifier=None,
) -> List[str]:
    """
    Processes the prompt to be used for the baseline inference model.
    :param prompt: the prompt to be processed.
    """
    # print(prompt)
    # tokens = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(tokens["input_ids"].to(
    # model.device), max_length=MAX_LENGTH, do_sample=True, num_return_sequences=num_generations)
    print(mode)
    # exit()
    len_prompt = len(tokenizer(prompt)["input_ids"])
    if mode is None:
        return pipe(
            prompt,
            max_length=max_length + len_prompt,
            do_sample=True,
            num_return_sequences=num_generations,
        )
    else:
        return pipe(
            prompt,
            max_length=max_length + len_prompt,
            do_sample=True,
            num_return_sequences=num_generations,
            control_vector=control_vector,
            control_coefficient=control_coefficient,
            control_threshold=control_threshold,
            mode=mode,
            eraser=eraser,
            proj=proj,
            Ws=Ws,
            classifier_linear_layer=tox_classifier,
        )


def process_entry(
    entry: dict,
    num_generations: int,
    tokenizer,
    model,
    pipe,
    max_length: int,
    control_vector=None,
    control_coefficient=None,
    control_threshold=None,
    mode="add",
    eraser=None,
    proj=None,
    Ws=None,
    tox_classifier=None,
) -> List[str]:
    """
    Processes the entry to be used for the baseline inference model.
    :param entry: the entry to be processed.
    """
    # print(entry)
    # print("out", eraser)
    prompt = entry["prompt"]["text"]
    return process_prompt(
        prompt,
        num_generations,
        tokenizer,
        model,
        pipe,
        max_length=max_length,
        control_vector=control_vector,
        control_coefficient=control_coefficient,
        control_threshold=control_threshold,
        mode=mode,
        eraser=eraser,
        proj=proj,
        Ws=Ws,
        tox_classifier=tox_classifier,
    )


def initialize_model_tok(model_id: str = DEFAULT_CHOICE, device: str = DEVICE_ID):
    """
    Initializes the baseline inference model.
    :param model_id: the model id to be used for the baseline inference model.
    """
    # model = GPT2LMHeadModel.from_pretrained(model_id)
    if model_id == "stanfordnlp/backpack-gpt2":
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = BackpackGPT2LMHeadModel.from_pretrained(
            model_id, config=config, trust_remote_code=True
        )
        tokenizer = GPT2Tokenizer.from_pretrained(DEFAULT_CHOICE)
    elif model_id == "custom_gpt":
        model = CustomGPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        model = GPT2LMHeadModel.from_pretrained(model_id)
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model.eval()
    model = model.to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer, pipe


DEFAULT_SETTINGS = {
    # format:
    # {model_id}: { setting }
    "custom_gpt": {
        "eraser_path": "eraser_af_gpt.pt",
        "eraser_fitter_path": "eraser_fitter_gpt_n.pt",
        "null_proj_path": "null_proj_clean_af_gpt.pt",
        "Ws_path": "null_proj_clean_Ws_af_gpt.pt",
        "row_proj": "null_proj_clean_row_proj_af_gpt.pt",
        "mean_diff_vec": "jigsaw_class_diff_vec.pt",
        "tox_classifier": "lrlayer_jigsaw_gpt.pt",
    },
    "backpack-gpt2": {
        "eraser_path": "eraser_af.pt",
        "eraser_fitter_path": "eraser_fitter_ad.pt",
        "null_proj_path": "null_proj_clean_af.pt",
        "Ws_path": "null_proj_clean_Ws_af.pt",
        "row_proj": "null_proj_clean_row_proj_af.pt",
        "control_vector": "trained_stuff/control_vector_clean.pt",
    },
    "gpt2": {
        "eraser_path": None,
        "eraser_fitter_path": None,
        "null_proj_path": None,
        "Ws_path": None,
        "row_proj": None,
        "control_vector": None,
    },
}


def create_output_file_name(
    model_id: str,
    dataset_path: str,
    control_threshold: Optional[float] = None,
    control_coefficient: Optional[float] = None,
    mode: str = "original",
    challenging: Optional[str] = None,
    left_layer: int = -1,
    intervention_artifact_path: Optional[str] = None,
    embedding_threshold: Optional[float] = None,
    embedding_coefficient: Optional[float] = None,
) -> str:
    path, extension = dataset_path.split(".")
    path_sep = path.split("/")

    ctrl_thresh_str, ctrl_coefficient_str = "", ""
    if control_threshold is not None:
        ctrl_thresh_str = str(control_threshold).replace(".", "_")
        ctrl_coefficient_str = str(control_coefficient).replace(".", "_")

    model_name_fr = model_id.split("/")[-1]

    if intervention_artifact_path is not None:
        intervention_artifact_path = intervention_artifact_path.split("/")[-1].replace(
            ".", ""
        )

    return (
        path_sep[-1]
        + f"c_thresh_{ctrl_thresh_str}_"
        + f"coeff_{ctrl_coefficient_str}_"
        + f"{mode}_"
        + f"{challenging}_{model_name_fr}_"
        + f"{left_layer if model_name_fr != 'backpack-gpt2' else ''}"
        + f"emb_{embedding_threshold if embedding_threshold is not None else ''}{embedding_coefficient if embedding_coefficient is not None else ''}"
        + f"{intervention_artifact_path if intervention_artifact_path is not None else ''}gen."
        + extension
    )


def main(
    model_id: str,
    num_generations: int = 5,
    max_length: int = MAX_LENGTH,
    file_name: str = "baseline_inference.txt",
    challenging: bool = True,
    control_coefficient: float = 0.0,
    control_threshold_percentile: Optional[float] = None,
    mode: str = "original",
    device: str = DEVICE_ID,
    dataset_files: Optional[List[str]] = None,
    non_challenging_ratio: Optional[float] = None,
    left_layer: int = -1,
    eraser_custom_path: Optional[str] = None,
    eraser_fitter_custom_path: Optional[str] = None,
    null_proj_custom_path: Optional[str] = None,
    embedding_threshold: Optional[float] = None,
    embedding_coefficient: Optional[float] = None,
    cons_opt_object_path: Optional[str] = None,
    lr_path: Optional[str] = None,
):
    """
    Runs the baseline inference model.
    :param model_id: the model id to be used for the baseline inference model.
    """
    model, tokenizer, pipe = initialize_model_tok(model_id, device=device)
    print(model.device, device)
    print(mode)

    control_vector, eraser, eraser_fitter, proj, Ws = None, None, None, None, None

    # paths to different learnt projections and erasures
    default_paths = DEFAULT_SETTINGS[model_id.split("/")[-1]]

    # overrides of default paths
    paths = default_paths
    if eraser_custom_path is not None:
        paths["eraser_path"] = eraser_custom_path
    if eraser_fitter_custom_path is not None:
        paths["eraser_fitter_path"] = eraser_fitter_custom_path
    if null_proj_custom_path is not None:
        paths["null_proj_path"] = null_proj_custom_path

    if mode in ["erase", "erase_better", "erase_baked"]:
        eraser = torch.load(open(paths["eraser_path"], "rb"), map_location=model.device)

    if mode in [
        "erase_fitter",
        "erase_fitter_baked",
        "erase_fitter_partial",
        "erase_fitter_explode",
        "cons_opt",
        "cons_opt_selective",
    ]:
        eraser_fitter = torch.load(
            open(paths["eraser_fitter_path"], "rb"), map_location=model.device
        )
        eraser = eraser_fitter
        print(paths["eraser_fitter_path"])

    if mode in ["nullp", "nullp_better", "null_proj_baked"]:
        proj = torch.load(
            open(paths["null_proj_path"], "rb"), map_location=model.device
        ).unsqueeze(dim=0)

    if mode in [
        "nullp_better",
        "erase_better",
        "null_proj_baked",
        "erase_baked",
        "erase_fitter_baked",
        "erase_fitter",
        "erase_fitter_partial",
        "erase_fitter_explode",
    ]:
        Ws = torch.load(open(paths["Ws_path"], "rb"), map_location=model.device)
        Ws_normed = [Ws[i] / torch.norm(Ws[i]) for i in range(len(Ws))]
        Ws = torch.stack(Ws_normed).squeeze().unsqueeze(dim=0)
    control_threshold = None

    # if not baked, we need to calculate the control threshold here
    # for baked, it is calculated inside the baking functions
    if mode not in ["erase_baked", "erase_fitter_baked", "null_proj_baked"]:
        if mode == "erase_fitter":
            if model_id == "stanfordnlp/backpack-gpt2":
                control_threshold = model.backpack.calculate_threshold_from_percentile(
                    eraser_fitter=eraser_fitter,
                    control_threshold_percentile=control_threshold_percentile,
                    to_apply_to="sense",
                ).item()
            elif model_id == "custom_gpt":
                raise NotImplementedError()
        elif mode == "nullp_better":
            if model_id == "stanfordnlp/backpack-gpt2":
                control_threshold = model.backpack.calculate_threshold_from_percentile(
                    Ws=Ws,
                    control_threshold_percentile=control_threshold_percentile,
                    to_apply_to="sense",
                ).item()
        elif mode == "erase_fitter_explode":
            control_threshold = control_threshold_percentile
    print("control_threshold", control_threshold)

    if mode in ["add", "mult", "erase", "boost", "nullp"]:
        control_vector = torch.load(
            open(paths["control_vector"], "rb"), map_location=model.device
        )

    if mode == "erase_baked":
        # currently not supported on GPT2
        Ws = Ws.squeeze()
        model.backpack.apply_transformation_to_sense(
            Ws,
            control_threshold_percentile,
            control_coefficient,
            eraser=eraser,
            null_proj=None,
        )
    elif mode == "null_proj_baked":
        # currently not supported on GPT2

        Ws = Ws.squeeze()

        if model_id == "custom_gpt":
            model.transformer.apply_transformation_to_mlp(
                proj=proj,
                control_threshold=control_threshold_percentile,
                control_coefficient=control_coefficient,
                layers=(left_layer, 0),
                Ws=Ws,
            )
        else:
            model.backpack.apply_transformation_to_sense(
                Ws,
                control_threshold_percentile,
                control_coefficient,
                eraser=None,
                null_proj=proj,
            )
    elif mode == "erase_fitter_baked":
        if model_id == "custom_gpt":
            layer_indices = -1
            if left_layer != -1:
                layer_indices = (left_layer, 0)
            model.transformer.apply_transformation_to_mlp(
                eraser_fitter=eraser_fitter,
                control_threshold=control_threshold_percentile,
                control_coefficient=control_coefficient,
                layers=layer_indices,
            )

            # do we also need to modify emebeddings?
            if embedding_threshold is not None:
                model.transformer.transform_embeddings(
                    eraser_fitter=eraser_fitter,
                    control_threshold=embedding_threshold,
                    control_coefficient=embedding_coefficient,
                )
        else:
            Ws = Ws.squeeze()
            model.backpack.apply_transformation_to_sense(
                Ws,
                control_threshold_percentile,
                control_coefficient,
                eraser_fitter=eraser_fitter,
                null_proj=None,
            )

    mean_diff_vec_path = None
    print(mode)
    tox_classifier = None
    if mode == "class_mean_diff":
        print("hi")
        mean_diff_vec_path = paths["mean_diff_vec"]
        control_vector = torch.load(
            open(mean_diff_vec_path, "rb"), map_location=model.device
        ).unsqueeze(dim=0)

        # load the state dict into torch linear layer
        classifier_path = paths["tox_classifier"]
        tox_classifier = torch.nn.Linear(768, 1)
        tox_classifier.load_state_dict(
            torch.load(open(classifier_path, "rb"), map_location=model.device)
        )
        tox_classifier.to(model.device)
        control_threshold = control_threshold_percentile
        print(tox_classifier)

    if mode == "cons_opt" or mode == "cons_opt_selective":
        print(mode)

        control_threshold = control_threshold_percentile
        contr_gen = ControlGeneration(eraser_fitter, device=model.device)
        linear_layer = contr_gen.make_linear_layer(control_coefficient)
        linear_layer.to(model.device)
        tox_classifier = linear_layer  # because I'm lazy

    if mode == "lr_arbit":
        print(mode)
        control_threshold = control_threshold_percentile
        linear_layer = torch.load(open(lr_path, "rb"), map_location=model.device)
        tox_classifier = linear_layer

    if dataset_files is None:
        dataset = initialize_data(challenging_only=challenging)
    else:
        dataset = []
        for file_path in dataset_files:
            dataset.append(
                initialize_data_from_json(
                    file_path,
                    challenging_only=challenging,
                    non_challenging_ratio=non_challenging_ratio,
                )
            )

    # num_generations = num_generations # lmaooooooooooooo, keeping this because its hilarious
    i = 0

    if dataset_files is None:
        with open(file_name, "w") as f:
            for entry in dataset:
                # print(entry)
                generations = process_entry(
                    entry,
                    num_generations,
                    tokenizer,
                    model,
                    pipe,
                    max_length=max_length,
                    control_vector=control_vector,
                    control_coefficient=control_coefficient,
                    control_threshold=control_threshold,
                    mode=mode,
                    eraser=eraser,
                    proj=proj,
                    Ws=Ws,
                )
                ret_entry = {}
                ret_entry.update(entry)
                ret_entry.update({"model_generations": generations})
                # print(ret_entry)
                f.write(json.dumps(ret_entry))
                f.write("\n")
                # i += 1100.0
    else:
        for i, df in enumerate(dataset):
            intervention_artifact_path = (
                eraser_custom_path or eraser_fitter_custom_path or null_proj_custom_path
            )
            if intervention_artifact_path is not None:
                intervention_artifact_path = intervention_artifact_path.split("/")[
                    -1
                ].replace(".", "")
            # if model != "custom_gpt":

            file_name = create_output_file_name(
                model_id=model_id,
                dataset_path=dataset_files[i],
                control_threshold=control_threshold_percentile,
                control_coefficient=control_coefficient,
                mode=mode,
                challenging=challenging,
                left_layer=left_layer,
                intervention_artifact_path=intervention_artifact_path,
                embedding_threshold=embedding_threshold,
                embedding_coefficient=embedding_coefficient,
            )

            print(file_name)

            # file_name = dataset_files[i].split(".")[0] + f"{challenging}_gen." + dataset_files[i].split(".")[1]
            with open(file_name, "w") as f:
                df_dict = df.to_dict(orient="records")
                for j, entry in enumerate(df_dict):
                    # print(entry)
                    # print("out", eraser)
                    generations = process_entry(
                        entry,
                        num_generations,
                        tokenizer,
                        model,
                        pipe,
                        max_length=max_length,
                        control_vector=control_vector,
                        control_coefficient=control_coefficient,
                        control_threshold=control_threshold,
                        mode=mode,
                        eraser=eraser,
                        proj=proj,
                        Ws=Ws,
                        tox_classifier=tox_classifier,
                    )
                    ret_entry = {}
                    ret_entry.update(entry)
                    ret_entry.update({"model_generations": generations})
                    # print(ret_entry)
                    f.write(json.dumps(ret_entry))
                    f.write("\n")
                    # i += 1100.0
                    # if j == 10:
                    # break


if __name__ == "__main__":
    # global DEVICE_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_CHOICE)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--device_id", type=int, default=DEVICE_ID)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    # parser.add_argument("--all", action="store_true")
    # parser.add_argument("--outfile", type=str,
    #                     default="baseline_inference.txt")
    # parser.add_argument("--custom_version", type=str, default=None)
    parser.add_argument("--control_coefficient", type=float, default=None)
    parser.add_argument("--control_threshold", type=float, default=None)
    parser.add_argument("--mode", type=str, default=None)
    # challenging, non, all
    parser.add_argument("--set", type=str, default="challenging")
    parser.add_argument("--dataset_files", type=str, nargs="+", default=None)
    parser.add_argument("--non_challenging_ratio", type=float, default=1.0)
    # range of layers to apply transformation
    parser.add_argument("--left_layer", type=int, default=-1)

    parser.add_argument("--eraser_custom_path", type=str, default=None)
    parser.add_argument("--eraser_fitter_custom_path", type=str, default=None)
    parser.add_argument("--null_proj_custom_path", type=str, default=None)

    parser.add_argument("--embedding_threshold", type=float, default=None)
    parser.add_argument("--embedding_coefficient", type=float, default=None)

    args = parser.parse_args()
    # outfile = args.model_id.split("/")[-1] + ".txt"
    DEVICE_ID = DEVICE + ":" + str(args.device_id)
    MAX_LENGTH = args.max_length

    out_file_name = f"coeff_{str(args.control_coefficient).replace('.', '_')}_cthresh_{str(args.control_threshold).replace('.', '_')}_gen_{args.mode}.txt"
    main(
        args.model_id,
        args.num_generations,
        max_length=args.max_length,
        file_name=out_file_name,
        control_coefficient=args.control_coefficient,
        control_threshold_percentile=args.control_threshold,
        mode=args.mode,
        challenging=args.set,
        device=DEVICE_ID,
        dataset_files=args.dataset_files,
        non_challenging_ratio=args.non_challenging_ratio,
        left_layer=args.left_layer,
        eraser_custom_path=args.eraser_custom_path,
        eraser_fitter_custom_path=args.eraser_fitter_custom_path,
        null_proj_custom_path=args.null_proj_custom_path,
        embedding_threshold=args.embedding_threshold,
        embedding_coefficient=args.embedding_coefficient,
    )
