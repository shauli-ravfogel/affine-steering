import pickle
import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression

def load_bios(model_name, layer, pooling, pca=False):
    if layer == -1:
       layer = "last"
    with open("../bios_data/bios_data/bios_train.pickle", "rb") as f:
        bios_train = pickle.load(f)

    with open("../bios_data/bios_data/bios_dev.pickle", "rb") as f:
        bios_dev = pickle.load(f)

    with open("../bios_data/bios_data/bios_test.pickle", "rb") as f:
        bios_test = pickle.load(f)


    text_train = [d["hard_text"] for d in bios_train]
    text_dev = [d["hard_text"] for d in bios_dev]
    text_test = [d["hard_text"] for d in bios_test]
    z_train = np.array([0 if d["g"] == "f" else 1 for d in bios_train])
    z_dev = np.array([0 if d["g"] == "f" else 1 for d in bios_dev])
    z_test = np.array([0 if d["g"] == "f" else 1 for d in bios_test])
    y_train = np.array([d["p"] for d in bios_train])
    y_dev =  np.array([d["p"] for d in bios_dev])
    y_test = np.array([d["p"] for d in bios_test])

    fname = "bios_encodings/bios_{}_{}_pooling:{}_layer:{}{}.npy"
    pca_str = "_pca" if pca else ""
    h_train = np.load(fname.format("train", model_name, pooling, layer, pca_str))
    h_dev = np.load(fname.format("dev", model_name, pooling, layer, pca_str))
    h_test = np.load(fname.format("test", model_name, pooling, layer, pca_str))

    assert len(y_train) == len(h_train)
    assert len(y_dev) == len(h_dev)
    assert len(y_test) == len(h_test)

    return text_train, text_dev, text_test, z_train, z_dev, z_test, y_train, y_dev, y_test, h_train, h_dev, h_test

def compute_tpr_gap(y_train, y_dev, z_train, z_dev, h_dev_transformed, h_dev, clf_before, clf_after):
    # Create mappings between labels and integers
    y2int = {y: i for i, y in enumerate(sorted(set(y_train)))}
    int2y = {i: y for y, i in y2int.items()}

    # Get predictions for the transformed and original classifiers
    y_pred = clf_after.predict(h_dev_transformed)
    #y_pred = np.array([int2y[i] for i in y_pred])

    y_pred_original = clf_before.predict(h_dev)
    #y_pred_original = np.array([int2y[i] for i in y_pred_original])

    y_pred_true = y_pred[y_pred == y_dev]
    z_pred_true = z_dev[y_pred == y_dev]

    rms_tpr_gap = 0.0
    rms_tpr_gap_orig = 0.0
    prof2tpr = {}
    prof2tpr_original = {}
    prof2percentfem = {}

    for y in set(y_pred_true):
        tpr_1 = (((y_pred[y_dev == y])[z_dev[y_dev == y] == 1]) == y).mean()
        tpr_0 = (((y_pred[y_dev == y])[z_dev[y_dev == y] == 0]) == y).mean()
        rms_tpr_gap += (tpr_1 - tpr_0) ** 2
        prof2tpr[y] = (tpr_1 - tpr_0) ** 2

        tpr_1_original = (((y_pred_original[y_dev == y])[z_dev[y_dev == y] == 1]) == y).mean()
        tpr_0_original = (((y_pred_original[y_dev == y])[z_dev[y_dev == y] == 0]) == y).mean()
        rms_tpr_gap_orig += (tpr_1_original - tpr_0_original) ** 2
        prof2tpr_original[y] = (tpr_1_original - tpr_0_original)

        prof2percentfem[y] = 1 - z_train[y_train == y].mean()

    rms_tpr_gap = np.sqrt(rms_tpr_gap / len(set(y_pred_true)))
    rms_tpr_gap_orig = np.sqrt(rms_tpr_gap_orig / len(set(y_pred_true)))

    return rms_tpr_gap, rms_tpr_gap_orig, prof2tpr, prof2tpr_original, prof2percentfem


def train_torch_logistic_classifier(x_train, y_train, batch_size=1024, num_epochs=100):

    # train a linear multiclass logistic regression (softmax) classifier
    num_classes = len(np.unique(y_train))
    clf = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], num_classes),
        torch.nn.Softmax(dim=1)
    )

    clf.to("cuda")
    x_train = torch.tensor(x_train).to("cuda").float()
    y2int = {y:i for i,y in enumerate(sorted(set(y_train)))}
    y_train = torch.tensor([y2int[yy] for yy in y_train]).to("cuda")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(clf.parameters())

    for i in tqdm.tqdm(range(num_epochs)):
        for j in range(0, len(x_train), batch_size):
            batch_idx = np.random.choice(len(x_train), batch_size)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            y_pred = clf(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return clf


def calc_accuracy(clf, x, y, device="cpu"):
    clf.to(device)
    x_torch = torch.tensor(x).to(device).float()
    y2int = {y:i for i,y in enumerate(sorted(set(y)))}
    y_int = torch.tensor([y2int[yy] for yy in y]).to("cpu")
    y_pred = clf(x_torch).cpu().argmax(axis=1)
    return (y_pred == y_int).float().mean()


def calc_accuracy_batched(clf, x, y, batch_size=32, device="cpu"):
    def to_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    clf.to(device)
    y2int = {yy: i for i, yy in enumerate(sorted(set(y)))}
    y_int = torch.tensor([y2int[yy] for yy in y]).to("cpu")

    correct = 0
    total = 0

    for x_batch, y_batch in zip(to_batches(x, batch_size), to_batches(y_int, batch_size)):
        x_torch = torch.tensor(x_batch).to(device).float()
        y_pred = clf(x_torch).cpu().argmax(axis=1)
        correct += (y_pred == y_batch).sum().item()
        total += y_batch.size(0)

    return correct / total

def get_predictions(clf, x):

    x_torch = torch.tensor(x).to("cuda").float()
    y_pred = clf(x_torch).cpu().argmax(axis=1)
    return y_pred.numpy()


def fit_logistic_regression(x_train,y_train,x_dev,y_dev,x_test,y_test, max_iter=30):
    import random

    random.seed(0)
    np.random.seed(0)

#     clf = SGDClassifier(loss="log", fit_intercept=True,  max_iter=3, tol = 0.1*1e-3,n_iter_no_change=1,
#                            n_jobs=32,alpha=1e-4)
    clf = LogisticRegression(warm_start = True, penalty = 'l2',
                        solver = "saga", multi_class = 'multinomial', fit_intercept = True,
                        verbose = 5, n_jobs = 64, random_state = 1, max_iter = max_iter)
    
    clf.fit(x_train, y_train)
    score_dev = clf.score(x_dev,y_dev)
    score_test = clf.score(x_test, y_test)
    
    return clf, score_dev, score_test



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
