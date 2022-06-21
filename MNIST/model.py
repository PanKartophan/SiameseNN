import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

        self.class_embeddings = nn.Parameter(torch.zeros(10, emb_dim))
        self.class_embeddings_num = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = F.normalize(x)
        return x

    def calc_class_embeddings(self, x, labels):
        embeddings = self.forward(x)
        for i in torch.unique(labels):
            self.class_embeddings[i] += embeddings[labels == i].sum(dim=0)
            self.class_embeddings_num[i] += (labels == i).sum()

    def predict(self, x):
        embeddings = self.forward(x)
        distances = torch.zeros(x.size(0), self.class_embeddings_num.size(0))
        for i in range(len(embeddings)):
            distances[i] = (self.class_embeddings / self.class_embeddings_num.unsqueeze(dim=1) -
                            embeddings[i].unsqueeze(dim=0)).pow(2).sum(dim=1)
        return torch.argmin(distances, dim=1)
