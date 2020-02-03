import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.PReLU(num_parameters=512),
            nn.Linear(in_features=512, out_features=1024),
            nn.PReLU(num_parameters=1024),
            nn.Linear(in_features=1024, out_features=out_features),
            nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features, out_features=256)
        self.relu1 = nn.PReLU(num_parameters=256)
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(in_features=256, out_features=512)
        self.relu2 = nn.PReLU(num_parameters=512)
        self.dropout2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(in_features=1, out_features=512)
        self.relu3 = nn.PReLU(num_parameters=512)
        self.dropout3 = nn.Dropout(0.2)

        self.linear4 = nn.Linear(in_features=512, out_features=out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = torch.max(x, dim=-1)[0]
        x = x.unsqueeze(dim=-1)

        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.linear4(x)

        y = self.sigmoid(x)

        return y
