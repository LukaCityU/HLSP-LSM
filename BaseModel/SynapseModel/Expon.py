import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Expon(nn.Module):
	def __init__(self, weight: np.ndarray, conn: np.ndarray, tau_syn=5., E=0., dt=1.):
		super().__init__()
		# Initial parameters
		self.in_dim = weight.shape[1]
		self.out_dim = weight.shape[0]

		self.register_buffer('E', torch.tensor(E).float())        # +/- 1 for excitatory/inhibitory synapse
		self.register_buffer('W', torch.tensor(weight).float())   # Synaptic weight (change in conductance)

		# Connection pattern mask, used to control weight updates;
		# set to 1 for connections and 0 for no connections.
		self.register_buffer('conn', torch.tensor(conn).float())

		self.tau_syn = tau_syn
		self.register_buffer('alpha', torch.exp(-dt / torch.tensor(tau_syn)))  # Decay coefficient of synaptic conductance

		# Initial variables
		self.memory_I = None
		self.memory_g = None
		self.device = None

	def forward(self, input_spike):
		"""
		Δg[t] = X * W
		g[t] = alpha * g[t-1] + Δg[t]
		I_syn[t] = g[t]

		:param input_spike: Presynaptic spike, shape = [batch, pre_size]
		:return: Postsynaptic current, shape = [batch, post_size]
		"""
		# （1）Compute conductance change Δg[t], shape=[post, size] (postsynaptic dim)
		delta_g = F.linear(input_spike, self.W)

		# （2）Update conductance: g[t] = alpha * g[t-1] + Δg[t]
		if self.memory_g is None:
			self.initialize_state(input_spike.shape[0], input_spike.device)
		g_t = self.alpha * self.memory_g + delta_g

		# （3）Compute synaptic current I_syn[t] = g[t]
		I_t = g_t * self.E

		# (4) Record current g and I_syn
		self.memory_g = g_t
		self.memory_I = I_t
		return I_t

	def initialize_state(self, batch_size, device):
		self.memory_I = torch.zeros(batch_size, self.out_dim).to(device)
		self.memory_g = torch.zeros(batch_size, self.out_dim).to(device)
		self.device = device

	def reset_state(self):
		self.memory_I = None
		self.memory_g = None
		self.device = None
