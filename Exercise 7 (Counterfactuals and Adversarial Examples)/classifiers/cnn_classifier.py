import torch
import torchvision as tv


class ImageNetClassifier(torch.nn.Module):

    def __init__(self) -> None:
        super(ImageNetClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
        )
        self.model.eval()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)

    def predict(self, x: torch.tensor, return_probs=False) -> torch.tensor:
        with torch.no_grad():
            output = self.model(x)
            probs = torch.nn.functional.softmax(output, -1)
            y_pred_prob, y_pred_idx = probs.max(1)
        return (y_pred_idx, y_pred_prob) if return_probs else y_pred_idx
