from concept_erasure import LeaceFitter
import torch
from counterfactuals.algos import fit_optimal_transport3


class CheapMeanShiftPostErasureInter:
    def __init__(self, eraser, shift, n, d, target_class: int = 0):
        self.eraser = eraser
        self.shift = shift
        self.target_class = target_class
        self.n = n
        self.d = d

    def __call__(self, X):
        return self.eraser(X) + self.shift.squeeze()

    def __str__(self) -> str:
        return f"erasure_shift_{self.target_class}"

    def report_dict(self):
        return {
            "target_class": self.target_class,
        }


class CheapMeanShiftPostErasureFitter:
    def __init__(
        self, X: torch.Tensor, Y: torch.Tensor, device: str, target_class: int = 0
    ):
        self.device = device
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.fitter = LeaceFitter.fit(X, Y)

        self.e_x = self.X.mean(dim=0).unsqueeze(1).to(self.device)

        # X | Y = 0 mean
        self.X_0 = self.X[self.Y == 0]
        self.mean_x_0 = self.X_0.mean(dim=0).unsqueeze(1)

        # X | Y = 1 mean
        self.X_1 = self.X[self.Y == 1]
        self.mean_x_1 = self.X_1.mean(dim=0).unsqueeze(1)

        # apply erasure to training
        eraser = self.fitter.eraser
        self.X_erased = eraser(self.X)
        self.erased_mean = self.X_erased.mean(dim=0).unsqueeze(1)
        self.target_class = target_class
        if target_class == 0:
            target_mean = self.mean_x_0
        else:
            target_mean = self.mean_x_1

        self.shift = target_mean - self.erased_mean

    def get_intervention(self):
        return CheapMeanShiftPostErasureInter(
            self.fitter.eraser,
            self.shift,
            n=self.n,
            d=self.d,
            target_class=self.target_class,
        )


class LeacePlusWassterSteinInter:
    def __init__(self, layer, eraser, d, target_class: int = 0):
        self.eraser = eraser
        self.layer = layer
        self.target_class = target_class
        self.d = d

    def __call__(self, X):
        return self.layer(self.eraser(X))

    def __str__(self):
        return f"leace_plus_wasster_stein_{self.target_class}"

    def report_dict(self):
        return {
            "target_class": self.target_class,
        }


class LeacePlusWassterStein:
    def __init__(
        self, X: torch.Tensor, Y: torch.Tensor, target_class: str, device: str
    ):
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.device = device

        self.fitter = LeaceFitter.fit(X, Y)
        self.eraser = self.fitter.eraser

        self.erased_X = self.eraser(self.X)

        self.X_given_0 = self.X[self.Y == 0]
        self.X_given_1 = self.X[self.Y == 1]

        self.target_class = target_class

        if target_class == 0:
            self.target_X = self.X_given_0
        if target_class == 1:
            self.target_X = self.X_given_1

        _, _, A = fit_optimal_transport3(
            train_x=self.erased_X.cpu().detach().numpy(),
            train_z=self.Y.cpu().detach().numpy(),
            dev_x=self.erased_X.cpu().detach().numpy(),
            target_x=self.target_X.cpu().detach().numpy(),
        )

        bias = self.target_X.mean(dim=0) - self.erased_X.mean(dim=0) @ torch.tensor(
            A, device=self.device, dtype=torch.float32
        )

        self.layer = torch.nn.Linear(self.d, self.d, bias=True).to(self.device)
        self.layer.weight.data = torch.tensor(
            A.T, device=self.device, dtype=torch.float32
        )
        self.layer.bias.data = bias

    def make_intervention(self):
        return LeacePlusWassterSteinInter(
            self.layer, self.eraser, d=self.d, target_class=self.target_class
        )
