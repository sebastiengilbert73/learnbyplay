import torch
import einops

class Usb(torch.nn.Module):
    def __init__(self, number_of_convs=64, latent_size=512, dropout_ratio=0.5):
        super(Usb, self).__init__()
        self.number_of_convs = number_of_convs
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(2, self.number_of_convs, kernel_size=(3, 3), padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.linear1 = torch.nn.Linear(self.number_of_convs * 4 * 5, self.latent_size)
        self.dropout1d = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(self.latent_size, 1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2, 6, 7)
        act1 = self.conv1(input_tsr)  # (N, C1, 4, 5)
        act2 = torch.nn.functional.relu(act1)  # (N, C1, 4, 5)
        act3 = self.dropout2d(act2)  # (N, C1, 4, 5)
        act4 = einops.rearrange(act3, 'n c h w -> n (c h w)', h=4, w=5)  # (N, C1 * 4 * 5)
        act5 = self.linear1(act4)  # (N, L)
        act6 = torch.nn.functional.relu(act5)  # (N, L)
        act7 = self.dropout1d(act6)  # (N, L)
        act8 = self.linear2(act7)

        return act8

class Dvi(torch.nn.Module):
    def __init__(self, nconv1=64, nconv2=128, latent_size=512, dropout_ratio=0.5):
        super(Dvi, self).__init__()
        self.nconv1 = nconv1
        self.nconv2 = nconv2
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(2, self.nconv1, kernel_size=(3, 3), padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.conv2 = torch.nn.Conv2d(self.nconv1, self.nconv2, kernel_size=(3, 3), padding=0)
        self.linear1 = torch.nn.Linear(self.nconv2 * 2 * 3, self.latent_size)
        self.dropout1d = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(self.latent_size, 1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2, 6, 7)
        act1 = self.conv1(input_tsr)  # (N, C1, 4, 5)
        act2 = torch.nn.functional.relu(act1)  # (N, C1, 4, 5)
        act3 = self.dropout2d(act2)  # (N, C1, 4, 5)
        act4 = self.conv2(act3)  # (N, C2, 2, 3)
        act5 = torch.nn.functional.relu(act4)  # (N, C2, 2, 3)
        act6 = einops.rearrange(act5, 'n c h w -> n (c h w)', h=2, w=3)  # (N, C2 * 2 * 3)
        act7 = self.linear1(act6)  # (N, L)
        act8 = torch.nn.functional.relu(act7)  # (N, L)
        act9 = self.dropout1d(act8)  # (N, L)
        act10 = self.linear2(act9)  # (N, 1)

        return act10

class Hdmi(torch.nn.Module):
    def __init__(self, nconv1=64, nconv2=128, latent_size=512, dropout_ratio=0.5):
        super(Hdmi, self).__init__()
        self.nconv1 = nconv1
        self.nconv2 = nconv2
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(2, self.nconv1, kernel_size=(4, 4), padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_ratio)
        self.conv2 = torch.nn.Conv2d(self.nconv1, self.nconv2, kernel_size=(3, 3), padding=0)
        self.linear1 = torch.nn.Linear(self.nconv2 * 1 * 2, self.latent_size)
        self.dropout1d = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(self.latent_size, 1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2, 6, 7)
        act1 = self.conv1(input_tsr)  # (N, C1, 3, 4)
        act2 = torch.nn.functional.relu(act1)  # (N, C1, 3, 4)
        act3 = self.dropout2d(act2)  # (N, C1, 3, 4)
        act4 = self.conv2(act3)  # (N, C2, 1, 2)
        act5 = torch.nn.functional.relu(act4)  # (N, C2, 1, 2)
        act6 = einops.rearrange(act5, 'n c h w -> n (c h w)', h=1, w=2)  # (N, C2 * 1 * 2)
        act7 = self.linear1(act6)  # (N, L)
        act8 = torch.nn.functional.relu(act7)  # (N, L)
        act9 = self.dropout1d(act8)  # (N, L)
        act10 = self.linear2(act9)  # (N, 1)

        return act10