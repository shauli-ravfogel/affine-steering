"""
This file has code to create and dispatch interventions based on the arguments
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .modify_model import WrapperModel
from .classify_linear_intervention import SelectiveIntervention

from collections import OrderedDict


def dispatcher(mode: str, device: str, **kwargs):
    if mode == "original":
        # identity transformation
        dims = kwargs.get("dims", None)
        if dims is None:
            print("Default value 768 taken because dims not mentioned")
            dims = 768
        weight_mat = torch.eye(dims, device=device)
        lin_lay = torch.nn.Linear(dims, dims, bias=True)
        lin_lay.weight.data = weight_mat
        lin_lay.bias.data = torch.zeros(dims, device=device)
        return lin_lay

    if mode == "classify_plus_apply":
        layer = kwargs.get("layer", None)
        classifier = kwargs.get("classifier", None)
        return SelectiveIntervention(layer=layer, classifier=classifier, device=device)


def initialize_model_tok(base_model: str, mode: str | None, device: str, **kwargs):
    tok = AutoTokenizer.from_pretrained(base_model)
    if mode == "original":
        model = AutoModelForCausalLM.from_pretrained(base_model).eval().to(device)
        pipe = pipeline(
            "text-generation", model=base_model, tokenizer=tok, device=device
        )
        return model, tok, pipe
    else:
        intervention = dispatcher(mode, device, **kwargs)
        model = (
            WrapperModel(
                base_model=base_model, intervention=intervention, generation=True
            )
            .eval()
            .to(device)
        )
        AutoModelForCausalLM.register(None, WrapperModel)
        pipe = pipeline("text-generation", model=model, tokenizer=tok, device=device)
        return model, tok, pipe
