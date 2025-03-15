import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor_fc1 = nn.Linear(input_dim, 512)
        self.actor_fc2 = nn.Linear(512, 256)
        self.actor_fc3 = nn.Linear(256, 128)
        self.actor_fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        a = F.relu(self.actor_fc3(a))
        return F.softmax(self.actor_fc4(a), dim=-1)

# Global model instance to avoid reloading on every call
_model = None

def load_policy(policy_filename):
    """Loads the policy model only once and returns it."""
    global _model
    if _model is None:
        _model = ActorCritic().to(device)
        _model.load_state_dict(torch.load(policy_filename, map_location=device))
        _model.eval()
    return _model

def policy_action(policy_filename, observation):
    """Predicts the action given an observation using the trained policy."""
    model = load_policy(policy_filename)
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = model(obs_tensor)
        return torch.argmax(action_probs).item()
