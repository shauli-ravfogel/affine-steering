
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cosine sim
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from sklearn.linear_model import SGDClassifier
# import mlp
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
# import pca
from sklearn.decomposition import PCA
import pickle
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pickle
import argparse

@torch.no_grad()
def encode(model, tokenizer, text, batch_size,layer=-1, pooling="last"):
  encodings = []

  with torch.no_grad():
   for i in tqdm.tqdm(range(0, len(text), batch_size)):
    batch = text[i:i+batch_size]
    padded_tokens = tokenizer(batch, padding=True, return_tensors="pt", max_length=128, truncation=True).to("cuda")
    outputs = model(**padded_tokens, output_hidden_states=True)
    lengths = padded_tokens["attention_mask"].sum(axis=1).detach().cpu().numpy()

    hiddens = outputs.hidden_states[layer]
    hiddens = hiddens.detach()
    for h,l in zip(hiddens, lengths):
      if pooling == "last":
        h = h[l-1]
      elif pooling == "cls":
        h = h[0]
      elif pooling == "mean":
        h = h[:l].mean(axis=0)
      encodings.append(h.detach().cpu().numpy())

  return np.array(encodings)


def load_bios():
    with open("../bios_data/bios_data/bios_train.pickle", "rb") as f:
        bios_train = pickle.load(f)

    with open("../bios_data/bios_data/bios_dev.pickle", "rb") as f:
        bios_dev = pickle.load(f)

    with open("../bios_data/bios_data/bios_test.pickle", "rb") as f:
        bios_test = pickle.load(f)


    text_train = [d["hard_text"] for d in bios_train]
    text_dev = [d["hard_text"] for d in bios_dev]
    text_test = [d["hard_text"] for d in bios_test]
    return text_train, text_dev, text_test

def load_tweets():
  texts = []
  for filename in "neg_neg.txt", "neg_pos.txt", "pos_neg.txt", "pos_pos.txt":
    with open("../tweets-data/"+filename, errors="replace") as f:
        text = [line.strip() for line in f.readlines()][:44000]
        texts.append(text)
  return texts




if __name__  == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
  parser.add_argument("--layer", type=int, default=-1)
  parser.add_argument("--pooling", type=str, default="last")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--data", type=str, default="bios")
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(args.model)

  if "bert" not in args.model:
    model = AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float16,
                                             device_map='auto').eval()
  else:
    model = AutoModel.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float16).eval().cuda()
  
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

  if args.data == "bios":
    text_train, text_dev, text_test = load_bios()
    data = [text_train, text_dev, text_test]
    filenames = ["train", "dev", "test"]
    output_path_template = "bios_encodings/bios_{}_{}_pooling:{}_layer:{}.npy"
  elif args.data == "tweets":
    data = load_tweets()
    filenames = ["neg_neg", "neg_pos", "pos_neg", "pos_pos"]
    output_path_template = "tweets_encodings/tweets_{}_{}_pooling:{}_layer:{}.npy"


  for text, filename in zip(data, filenames):
    
    encodings = encode(model, tokenizer, text, args.batch_size, layer=args.layer, pooling=args.pooling)
    layer = "last" if args.layer == -1 else args.layer
    np.save(output_path_template.format(filename, args.model.split("/")[1], args.pooling, layer), encodings)

