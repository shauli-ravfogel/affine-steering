import torch
from collections import OrderedDict


class SelectiveIntervention:
    """
    takes a classifier and a pytorch layer
    classifier is a pytorch model d -> 1
    layer is a pytorch layer that preserves the shape
    """

    def __init__(self, layer, classifier=None, device: str = "cpu"):
        self.device = device

        if isinstance(layer, str):
            self.layer_path = layer
            self.layer = torch.load(layer, map_location=self.device)
        else:
            self.layer_path = None
            self.layer = layer

        self.classifier_path = classifier

        self.classifier = None
        if classifier is not None:
            classifier = torch.load(classifier, map_location=self.device)
            if isinstance(classifier, OrderedDict):
                # is a state dict, we assume linear layer
                in_features = classifier["weight"].shape[1]
                out_features = classifier["weight"].shape[0]
                self.classifier = None
                if "bias" in classifier:
                    self.classifier = torch.nn.Linear(
                        in_features, out_features, bias=True
                    )
                else:
                    self.classifier = torch.nn.Linear(
                        in_features, out_features, bias=False
                    )
                self.classifier.load_state_dict(classifier)
                # put a sigmoid over the classifier
                self.classifier = torch.nn.Sequential(
                    self.classifier, torch.nn.Sigmoid()
                )
            else:
                self.classifier = classifier
            self.classifier.to(device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        if a classifier is sent then use it, else don't
        """

        if self.classifier is None:
            return self.layer(x)
        else:
            mask = torch.where(self.classifier(x) > 0.5, 1, 0)
            inv_mask = torch.where(mask == 0, 1, 0)
            og = x
            transformed = self.layer(x)

            return og * inv_mask + transformed * mask

    def report_dict(self):
        if self.layer_path is not None:
            return {
                "layer": self.layer_path,
                "classifier": self.classifier_path,
            }
        else:
            d = self.layer.report_dict()
            d["classifier"] = self.classifier_path
            return d

    def __str__(self) -> str:
        layer_path = (
            self.layer_path.split("/")[-1].split(".")[0]
            if self.layer_path is not None
            else None
        )
        classifier_path = (
            self.classifier_path.split("/")[-1].split(".")[0]
            if self.classifier_path is not None
            else None
        )
        if layer_path is None:
            layer_path = self.layer.__str__()
        return f"classify_plus_apply_{layer_path}_{classifier_path}_"
