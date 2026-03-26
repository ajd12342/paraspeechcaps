# Adapted from the VoxSim trainer repository by Ahn et al.
# https://github.com/kaistmm/voxsim_trainer
# Ahn et al., "VoxSim: A Perceptual Voice Similarity Dataset", Interspeech 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.criterion = torch.nn.MSELoss()
        print('Initialised MSE Loss')

    def forward(self, x, label=None):
        embed1 = x[:, 0, :]
        embed2 = torch.mean(x[:, 1:, :], 1)
        score = F.cosine_similarity(embed1, embed2)
        nloss = self.criterion(score, label)
        return nloss, score
