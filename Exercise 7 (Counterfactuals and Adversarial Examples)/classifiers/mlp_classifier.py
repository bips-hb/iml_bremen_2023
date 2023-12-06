import torch
import torch.nn as nn
from pathlib import Path

WEIGHTS_FILE = Path('weights.pt')


class MLPClassifier(torch.nn.Module):

    def __init__(self, num_input_features, num_classes, num_hidden) -> None:
        super(MLPClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(num_input_features, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_classes)
        )
        self.model.eval()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)

    def fit(self, train_dataset, eval_dataset, epochs=10):
        self.train()
        dl = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(1, epochs+1):
            for batch in dl:
                x, y = batch
                self.zero_grad()
                outputs = self(x)
                loss = nn.functional.cross_entropy(outputs, y.long())
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            dl = torch.utils.data.DataLoader(eval_dataset, batch_size=8)
            targets = []
            preds = []
            losses = []
            for batch in dl:
                x, y = batch
                outputs = self(x)
                loss = nn.functional.cross_entropy(outputs, y.long())
                
                targets.extend(y)
                preds.extend(outputs.argmax(-1))
                losses.append(loss)
            accuracy = (torch.tensor(targets) == torch.tensor(preds)).sum() / len(eval_dataset)
            avg_loss = torch.tensor(losses).mean()
            print(f'Test loss: {avg_loss:.2}\tTest accuracy: {accuracy:.2}')

    def predict(self, x: torch.tensor, return_probs=False) -> torch.tensor:
        with torch.no_grad():
            output = self.model(x)
            probs = torch.nn.functional.softmax(output, -1)
            y_pred_prob, y_pred_idx = probs.max(1)
        return (y_pred_idx, y_pred_prob) if return_probs else y_pred_idx


