import torch

class SaintAndre(torch.nn.Module):
    def __init__(self, latent_size=512, dropout_ratio=0.5):
        super(SaintAndre, self).__init__()
        self.latent_size = latent_size

        self.linear1 = torch.nn.Linear(2 * 3 * 3, self.latent_size)
        self.linear2 = torch.nn.Linear(self.latent_size, 1)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2 * 3 * 3)
        act1 = self.linear1(input_tsr)  # (N, L)
        act2 = torch.nn.functional.relu(act1)  # (N, L)
        act3 = self.dropout(act2)  # (N, L)
        act4 = self.linear2(act3)  # (N, 1)
        return act4
