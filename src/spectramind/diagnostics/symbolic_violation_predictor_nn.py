"""
Neural network version of symbolic violation predictor.
Trains on μ spectra to predict symbolic violation likelihood.
"""
import torch
import torch.nn as nn

class SymbolicViolationPredictorNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rules):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_rules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))
