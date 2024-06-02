import pandas as pd
import torch
import numpy as np
from concept_erasure import LeaceFitter
from counterfactuals.algos import fit_optimal_transport, fit_optimal_transport2
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def balance_dataset(df: pd.DataFrame):
    """
    Balance the dataset by under-sampling the majority
    """
    # df["toxic"] = df["target"] >= 0.5
    # df["toxic"] = df["toxic"].astype(int)

    toxic_split = df[df["toxic"] == 1]
    non_toxic_split = df[df["toxic"] == 0]

    non_toxic_split_sampled = non_toxic_split.sample(
        n=len(toxic_split), random_state=42
    )

    balanced_split = pd.concat([toxic_split, non_toxic_split_sampled])

    # drop all rows where 'comment_text' is None
    balanced_split = balanced_split.dropna(subset=["comment_text"])
    train_split, test_split = train_test_split(
        balanced_split, test_size=0.3, random_state=42
    )
    return (train_split, test_split)


def get_representations_jigsaw_spec_layer(
    data: pd.DataFrame, model, layer_num, tokenizer, device, num: int
):
    """
    Get next token pred representations for every prompt in data
    """
    all_states = []
    if num == -1:
        num = len(data)
    for i in range(num):
        prompt = data.iloc[i]["comment_text"]
        # print(prompt)
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_tokens = prompt_tokens[:, :512]
        # print(prompt_tokens)
        with torch.no_grad():
            hiddens = model(input_ids=prompt_tokens, position_ids=None).hidden_states
            # print(hidden_states.shape)
            if type(hiddens) != tuple:  # backpacks
                prompt_rep = hiddens[:, -1, :]
            else:  # gpt2
                prompt_rep = hiddens[layer_num][:, -1, :]
            # print(model(input_ids=prompt_tokens, position_ids=None).hidden_states.shape)
        all_states.append(prompt_rep.squeeze().cpu())
    return torch.stack(all_states)


def get_representations_jigsaw(data: pd.DataFrame, model, tokenizer, device, num: int):
    """
    Get next token pred representations for every prompt in data
    """
    all_states = []
    if num == -1:
        num = len(data)
    for i in range(num):
        prompt = data.iloc[i]["comment_text"]
        # print(prompt)
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_tokens = prompt_tokens[:, :512]
        # print(prompt_tokens)
        with torch.no_grad():
            hiddens = model(input_ids=prompt_tokens, position_ids=None).hidden_states
            # print(hidden_states.shape)
            if type(hiddens) != tuple:  # backpacks
                prompt_rep = hiddens[:, -1, :]
            else:  # gpt2
                prompt_rep = hiddens[-1][:, -1, :]
            # print(model(input_ids=prompt_tokens, position_ids=None).hidden_states.shape)
        all_states.append(prompt_rep.squeeze().cpu())
    return torch.stack(all_states)


def fit_and_save_leace(
    model_id: str, X, Y, dataset_name: str, percentage_dataset: float, save: bool = True
):
    """
    Fit and save a LeaceFitter object
    """
    leace_fitter = LeaceFitter.fit(X, Y)
    if save:
        torch.save(
            leace_fitter,
            f"artifacts/leace_fitter_{model_id}_{dataset_name}_{percentage_dataset}.pt",
        )
    else:
        return leace_fitter


def save_wasser_stein_mapping_linear_layer(
    model_id: str, model, X, Y, dataset_name, percentage_dataset, save: bool = True
):
    inv_labels = 1 - Y
    _, _, A = fit_optimal_transport(
        X.cpu().detach().numpy(),
        inv_labels,
        X.cpu().detach().numpy(),
        inv_labels,
    )

    X_given_0 = X[Y == 0]
    X_given_1 = X[Y == 1]
    mu_0 = X_given_0.mean(dim=0)
    mu_1 = X_given_1.mean(dim=0)

    bias = mu_0 - mu_1 @ torch.tensor(A, device=X.device, dtype=torch.float32)

    gaussian_transfer_layer = torch.nn.Linear(
        model.config.n_embd, model.config.n_embd, bias=True
    )
    gaussian_transfer_layer.weight.data = torch.tensor(
        A, device=X.device, dtype=torch.float32
    ).T
    gaussian_transfer_layer.bias.data = bias

    if save:
        torch.save(
            gaussian_transfer_layer,
            f"artifacts/gaussian_transfer_layer_0_to_1{model_id}_{dataset_name}_{percentage_dataset}.pt",
        )
    else:
        return gaussian_transfer_layer


def save_classifier_linear_layer(
    model_id: str, model, X, Y, dataset_name, percentage_dataset, save: bool = True
):
    """
    Fit and save a linear layer classifier
    """
    clf = LogisticRegression(max_iter=4000, fit_intercept=True)
    clf.fit(X.cpu().detach().numpy(), Y.cpu().detach().numpy())

    vec_coef_ = torch.tensor(clf.coef_, device=X.device, dtype=torch.float32)
    vec_intercept_ = torch.tensor(clf.intercept_, device=X.device, dtype=torch.float32)
    layer = torch.nn.Linear(model.config.n_embd, 1, bias=True)
    layer.weight.data = vec_coef_
    layer.bias.data = vec_intercept_

    if save:
        torch.save(
            layer, f"artifacts/clf_{model_id}_{dataset_name}_{percentage_dataset}.pt"
        )
    else:
        return layer


from sklearn.neural_network import MLPClassifier
import sklearn


def fit_run_mlp(X, Y, X_test, Y_test, device, num_epochs=1000, lr=0.001):
    mlp = MLPClassifier(
        hidden_layer_sizes=[500], max_iter=num_epochs, learning_rate_init=lr
    )
    mlp.fit(X.cpu().detach().numpy(), Y.cpu().detach().numpy())
    mlp_preds = mlp.predict(X_test.cpu().detach().numpy())
    classification_report = sklearn.metrics.classification_report(
        Y_test.cpu().detach().numpy(), mlp_preds, output_dict=True
    )
    return {
        "mlp": mlp,
        "classification_report": classification_report,
    }


import pickle


def main(
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    device: str,
    percentage_dataset: float = 1.0,
):
    # initialize model
    model_gpt = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True
    )
    tokenizer_gpt = AutoTokenizer.from_pretrained(model_id)
    model_gpt.eval()
    model_gpt.to(device)
    print("Models initialized")
    # load dataset
    df = pd.read_csv(dataset_path)
    df_train, df_test = balance_dataset(df)
    len_df = len(df)

    if percentage_dataset < 1.0:
        df_train = df_train.sample(frac=percentage_dataset, random_state=42)
        df_test = df_test.sample(frac=percentage_dataset, random_state=42)
        print(f"Using {percentage_dataset} of the dataset")

    Y, Y_test = (
        torch.tensor(df_train["toxic"].values, device="cpu"),
        torch.tensor(df_test["toxic"].values, device="cpu"),
    )

    print("Dataset loaded")

    n_layers = model_gpt.config.n_layer

    reports = []
    leaces = []
    mlps = []
    ot_maps = []
    lrs = []

    with torch.no_grad():
        for cur_layer in range(n_layers):
            # get representations
            X = get_representations_jigsaw_spec_layer(
                df_train, model_gpt, cur_layer, tokenizer_gpt, device, num=-1
            )
            X_test = get_representations_jigsaw_spec_layer(
                df_test, model_gpt, cur_layer, tokenizer_gpt, device, num=-1
            )

            # fit LEACE
            leace = fit_and_save_leace(
                model_id, X, Y, dataset_name, percentage_dataset, save=False
            )
            leaces.append(leace)

            # fit wasserstein mapping
            ot_map = save_wasser_stein_mapping_linear_layer(
                model_id, model_gpt, X, Y, dataset_name, percentage_dataset, save=False
            )
            ot_maps.append(ot_map)

            # fit classifier
            lr = save_classifier_linear_layer(
                model_id, model_gpt, X, Y, dataset_name, percentage_dataset, save=False
            )
            lrs.append(lr)

            # fit MLP
            mlp_and_report = fit_run_mlp(
                X, Y, X_test, Y_test, device, num_epochs=5000, lr=0.001
            )
            mlp = mlp_and_report["mlp"]
            mlp_report = mlp_and_report["classification_report"]
            mlps.append(mlp)
            reports.append(mlp_report)

    # save reports
    with open(
        f"mult_layer_runs_and_artifacts/mlp_reports_{model_id}_{dataset_name}_{percentage_dataset}.pkl",
        "wb",
    ) as f:
        pickle.dump(reports, f)
    # save leaces
    with open(
        f"mult_layer_runs_and_artifacts/leaces_{model_id}_{dataset_name}_{percentage_dataset}.pkl",
        "wb",
    ) as f:
        pickle.dump(leaces, f)
    # save mlps
    with open(
        f"mult_layer_runs_and_artifacts/mlps_{model_id}_{dataset_name}_{percentage_dataset}.pkl",
        "wb",
    ) as f:
        pickle.dump(mlps, f)
    # save ot_maps
    torch.save(
        ot_maps,
        f"mult_layer_runs_and_artifacts/ot_maps_{model_id}_{dataset_name}_{percentage_dataset}.pt",
    )

    # save lrs
    torch.save(
        lrs,
        f"mult_layer_runs_and_artifacts/lrs_{model_id}_{dataset_name}_{percentage_dataset}.pt",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="model id",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="path to dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="dataset name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device",
    )

    parser.add_argument(
        "--percentage_dataset",
        type=float,
        default=1.0,
        help="percentage of dataset to use",
    )

    args = parser.parse_args()

    main(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        device=args.device,
        percentage_dataset=args.percentage_dataset,
    )
