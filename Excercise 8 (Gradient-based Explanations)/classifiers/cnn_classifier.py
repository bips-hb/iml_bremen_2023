import torch
import torchvision as tv
from torchvision.models.resnet import ResNet18_Weights


class ImageNetClassifier(torch.nn.Module):

    def __init__(self) -> None:
        super(ImageNetClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            tv.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
            torch.nn.Softmax(1)
        )
        self.model.eval()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)

    def predict(self, x: torch.tensor, return_probs=False) -> torch.tensor:
        with torch.no_grad():
            output = self.model(x)
            y_pred_prob, y_pred_idx = output.max(1)
        return (y_pred_idx, y_pred_prob) if return_probs else y_pred_idx
