import torch
import torch.nn as nn

class Critic(nn.Module):
	def __init__(self,state_dim, action_dim):
		super(Critic, self).__init__()

		offset = 0

		# Critic 1
		self.C1 = nn.Sequential(nn.Linear(state_dim+action_dim+offset,512),
										nn.ReLU(),
										nn.Linear(512,1024),
										nn.ReLU(),
										nn.Linear(1024,512),
										nn.ReLU(),
										nn.Linear(512,1)) 

		# Critic 2
		self.C2 = nn.Sequential(nn.Linear(state_dim+action_dim+offset,512),
										nn.ReLU(),
										nn.Linear(512,1024),
										nn.ReLU(),
										nn.Linear(1024,512),
										nn.ReLU(),
										nn.Linear(512,1)) 		


	def forward(self,state,action):

		input = torch.cat([state,action],1)

		q1, q2 = self.C1(input), self.C2(input)

		return q1, q2



	def Q1(self,state,action):

		input = torch.cat([state,action],1)

		q1 = self.C1(input)

		return q1

