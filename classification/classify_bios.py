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
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import pickle
import torch
import tqdm
import ot
import argparse
import utils
import leace

def perform_pca(h_train, h_dev, h_test, pca_dim=768):
   
    pca = PCA(n_components=pca_dim, random_state=0)
    pca.fit(h_train)

    h_train_pca = pca.transform(h_train)
    h_dev_pca = pca.transform(h_dev)
    h_test_pca = pca.transform(h_test)
    return h_train_pca, h_dev_pca, h_test_pca



def apply_steering(mlp, fitted_ot, h_train, h_dev, h_test, z_labels_train):
    z_train_pred = mlp.predict(h_train)
    z_dev_pred = mlp.predict(h_dev)
    z_test_pred = mlp.predict(h_test)

    x_dev_source = h_dev[z_dev_pred==0]
    x_dev_target = h_dev[z_dev_pred==1]
    x_test_source = h_test[z_test_pred==0]
    x_test_target = h_test[z_test_pred==1]

    train_x_transformed = h_train.copy()
    dev_x_transformed = h_dev.copy()
    test_x_transformed = h_test.copy()

    # Mean+Covariance matching

    x_train_source = h_train[z_labels_train==0]
    x_train_target = h_train[z_labels_train==1]
    train_x_transformed[z_labels_train==0] = fitted_ot.transform(Xs=x_train_source)
    
    dev_x_transformed[z_dev_pred==0] = fitted_ot.transform(Xs=x_dev_source)
    test_x_transformed[z_test_pred==0] = fitted_ot.transform(Xs=x_test_source)

    # Mean Matching

    mean_diff_vec = x_train_target.mean(axis=0) - x_train_source.mean(axis=0)
    train_x_transformed_steering = h_train.copy()
    dev_x_transformed_steering = h_dev.copy()
    test_x_transformed_steering = h_test.copy()

    train_x_transformed_steering[z_labels_train==0] += mean_diff_vec
    dev_x_transformed_steering[z_dev_pred==0] += mean_diff_vec
    test_x_transformed_steering[z_test_pred==0] += mean_diff_vec

    return (train_x_transformed, dev_x_transformed, test_x_transformed), (train_x_transformed_steering, dev_x_transformed_steering, test_x_transformed_steering)

def apply_leace(h_train, h_dev, h_test, z_labels_train):

    eraser = leace.LeaceEraser.fit(torch.tensor(h_train).float(), torch.tensor(z_labels_train))
    h_train_leace = eraser(torch.tensor(h_train).float())
    h_dev_leace = eraser(torch.tensor(h_dev).float())
    h_test_leace = eraser(torch.tensor(h_test).float())

    return (h_train_leace, h_dev_leace, h_test_leace)




  
if __name__  == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
  parser.add_argument("--layer", type=str, default="last")
  parser.add_argument("--do_pca", type=int, default=1)
  parser.add_argument("--pooling", type=str, default="last")
  args = parser.parse_args()

  np.random.seed(0)
  torch.manual_seed(0)
  print("Loading data...")
  text_train, text_dev, text_test, z_train, z_dev, z_test, y_train, y_dev, y_test, h_train, h_dev, h_test = utils.load_bios(args.model, args.layer, args.pooling)
  if args.do_pca:
    print("Applying PCA...")
    h_train, h_dev, h_test = perform_pca(h_train, h_dev, h_test, pca_dim=768)
    # save
    np.save("bios_encodings/bios_train_{}_pooling:{}_layer:{}_pca.npy".format(args.model, args.pooling, args.layer), h_train)
    np.save("bios_encodings/bios_dev_{}_pooling:{}_layer:{}_pca.npy".format(args.model, args.pooling, args.layer), h_dev)
    np.save("bios_encodings/bios_test_{}_pooling:{}_layer:{}_pca.npy".format(args.model, args.pooling, args.layer), h_test)

  # fit an MLP to predict gener labels & fit OLC (mean+covariance matching)
  print("Fitting MLP...")
  mlp = MLPClassifier(hidden_layer_sizes=(128), early_stopping=True)
  mlp.fit(h_train, z_train)
  print("MLP accuracy: {:.3f}".format(mlp.score(h_dev, z_dev)))

  z_dev_pred = mlp.predict(h_dev)
  x_train_source = h_train[z_train==0]
  x_train_target = h_train[z_train==1]
  print("Fitting MiMiC...")
  ot_linear = ot.da.LinearTransport(reg=1e-5)
  ot_linear.fit(Xs=x_train_source, Xt=x_train_target)

  # save mlp and ot

  with open("interim/mlp_ot_bios_{}_pooling:{}_layer:{}.pickle".format(args.model, args.pooling, args.layer), "wb") as f:
    pickle.dump({"mlp": mlp, "ot": ot_linear}, f)
    
  # apply steering

  (h_train_transformed, h_dev_transformed, h_test_transformed),(h_train_transformed_steering, h_dev_transformed_steering, h_test_transformed_steering) = apply_steering(mlp, ot_linear, h_train, h_dev, h_test, z_train)
  h_train_leace, h_dev_leace, h_test_leace = apply_leace(h_train, h_dev, h_test, z_train)
  # save vectors in a dict in the interim dir

  with open("interim/bios_transformed_{}_pooling:{}_layer:{}.pickle".format(args.model, args.pooling, args.layer), "wb") as f:
    pickle.dump({"train": h_train_transformed, "dev": h_dev_transformed, "test": h_test_transformed}, f)
  with open("interim/bios_transformed_steering_{}_pooling:{}_layer:{}.pickle".format(args.model, args.pooling, args.layer), "wb") as f:
    pickle.dump({"train": h_train_transformed_steering, "dev": h_dev_transformed_steering, "test": h_test_transformed_steering}, f)
    with open("interim/bios_leace_{}_pooling:{}_layer:{}.pickle".format(args.model, args.pooling, args.layer), "wb") as f:
        pickle.dump({"train": h_train_leace, "dev": h_dev_leace, "test": h_test_leace}, f)

  # train classifiers
  print("Training profession classifiers...")
  clf_before, acc_before, _ = utils.fit_logistic_regression(h_train, y_train, h_dev, y_dev, h_test, y_test)
  print("Accuracy before steering: {:.3f}".format(acc_before))
  clf_after_steering, acc_after_steering, _ = utils.fit_logistic_regression(h_train_transformed_steering, y_train, h_dev_transformed_steering, y_dev, h_test_transformed_steering, y_test)
  print("Accuracy after steering (Mean Matching): {:.3f}".format(acc_after_steering))
  clf_after, acc_after, _ = utils.fit_logistic_regression(h_train_transformed, y_train, h_dev_transformed, y_dev, h_test_transformed, y_test)
  print("Accuracy after steering (Mean+Covariance Matching): {:.3f}".format(acc_after))
  clf_after_leace, acc_after_leace, _ = utils.fit_logistic_regression(h_train_leace, y_train, h_dev_leace, y_dev, h_test_leace, y_test)
  print("Accuracy after steering (LEACE): {:.3f}".format(acc_after_leace))
  
#   clf_before = utils.train_torch_logistic_classifier(h_train, y_train)
#   acc_before = utils.calc_accuracy(clf_before, h_dev, y_dev)

#   clf_after = utils.train_torch_logistic_classifier(h_train_transformed, y_train)
#   acc_after = utils.calc_accuracy(clf_after, h_dev_transformed, y_dev)

#   clf_after_steering = utils.train_torch_logistic_classifier(h_train_transformed_steering, y_train)
#   acc_after_steering = utils.calc_accuracy(clf_after_steering, h_dev_transformed_steering, y_dev)

  # save clfs

  for clf, clf_name in zip([clf_before, clf_after, clf_after_steering, clf_after_leace], ["before", "after", "after_steering", "after_leace"]):
    with open("interim/clf_bios_{}_pooling:{}_layer:{}_{}.pickle".format(args.model, args.pooling, args.layer, clf_name), "wb") as f:
        pickle.dump(clf, f)
