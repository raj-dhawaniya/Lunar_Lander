import numpy as np
import torch
import torch.nn as nn

def policy_action(params, observation):
    # Check if params is a PyTorch state_dict (Neural Network model) or raw weights (Genetic Algorithm)
    if isinstance(params, dict):
        model = PolicyNetwork()
        model.load_state_dict(params)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(observation, dtype=torch.float32))
        return torch.argmax(logits).item()
    else:
        W = params[:32].reshape(8, 4)
        b = params[32:].reshape(4)
        logits = np.dot(observation, W) + b
        return np.argmax(logits)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)  # First layer (8 inputs → 16 neurons)
        self.fc2 = nn.Linear(16, 4)  # Second layer (16 neurons → 4 outputs)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output logits
