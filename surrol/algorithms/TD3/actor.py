import torch
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		
		offset = 0
		
		l1 = nn.Linear(state_dim+offset, 512)
		l2 = nn.Linear(512, 1024)
		l3 = nn.Linear(1024, 512)
		l4 = nn.Linear(512, action_dim)

		self.A = nn.Sequential(l1,nn.ReLU(),l2,nn.ReLU(),l3,nn.ReLU(),l4)
		
		self.max_action = max_action
		
	def forward(self, state):
		
		out = self.A(state)
		
		return self.max_action * torch.tanh(out)