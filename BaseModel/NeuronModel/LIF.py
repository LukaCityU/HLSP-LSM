"""
LIF Model:
	tau_m * (dV / dt) = -V(t) + RI(t)
Difference Equation (R=1):
	V[t] = beta * V[t-1] + (1-beta) * I[t]
Exponential Euler:
	beta = exp(-dt/tau_m)
	V[t] = exp(-dt/tau_m) * V[t-1] + (1-exp(-dt/tau_m)) * I[t]
Engineering Simplified Form：
	V[t] = exp(-dt/tau_m) * V[t-1] + I[t]
"""
import torch
import torch.nn as nn
import numpy as np


# noinspection PyTypeChecker
class LIF(nn.Module):
	def __init__(self, size: int, V_init: np.ndarray, V_reset=13.5, V_th=15., tau_m=30., t_ref=3., R=1., dt=1.):
		"""
		Ensure unit consistency during computation:
		voltage (mV), resistance (MΩ), time constant (ms), conductance (μS), current (nA), capacitance (nF)

		:param size: Number of neuron
		:param V_init: Initial membrane potential, shape=[size, ]
		:param V_reset: Reset potential
		:param V_th: Firing threshold
		:param tau_m: Membrane time constant
		:param t_ref: Refractory period
		:param R: Membrane resistance
		:param dt: Integration timestep
		"""
		super().__init__()
		# Initial parameters
		self.size = size
		self.register_buffer('V_reset', torch.tensor(V_reset).float())
		self.register_buffer('V_th', torch.tensor(V_th).float())
		self.register_buffer('tau_m', torch.tensor(tau_m).float())
		self.register_buffer('t_ref', torch.tensor(t_ref).float())
		self.register_buffer('R', torch.tensor(R).float())
		self.register_buffer('dt', torch.tensor(dt).float())
		self.register_buffer('beta1', torch.exp(-self.dt / self.tau_m))
		self.register_buffer('beta2', 1. - torch.exp(-self.dt / self.tau_m))
		self.register_buffer('V_init', torch.tensor(V_init).float())

		# Initial variables
		self.device = None
		self.memory_voltage = None  # t-1 Membrane potential
		self.memory_spike = None    # t-1 Spike
		self.sum_spike = None       # Total spike count
		self.ref_th = None          # Refractory period threshold
		self.ref = None             # Initial Refractory period

	def forward(self, I):
		"""
		Compute membrane potential, V[t] --> Generate spike, S[t] --> Reset membrane potential, V[t] = V_reset

		:param I: total input current, shape: [batch, self.size]

		:return:
		"""
		if self.memory_voltage is None:
			self.initialize_state(I.shape[0], I.device)

		# V[t]
		# t_voltage = (self.beta1 * self.memory_voltage) + (self.beta2 * self.R * I * self.tau_m)
		t_voltage = (self.beta1 * self.memory_voltage) + (self.R * I)

		# Neurons with refractory time below threshold stay at the reset potential,
		# and only neurons above the threshold accumulate membrane potential
		cant_spike = self.ref < self.ref_th
		t_voltage = torch.where(cant_spike, self.memory_voltage, t_voltage)

		# S[t]
		t_spike = t_voltage.ge(self.V_th).float()  # ge: >=

		# Reset flag: neurons that have fired must have their V[t] and ref[t] reset to 0
		flag_reset = (t_voltage >= self.V_th)
		t_voltage[flag_reset] = self.V_reset
		self.ref += 1
		self.ref[flag_reset] = 0

		# Record data at the current time step
		self.memory_voltage = t_voltage
		self.memory_spike = t_spike
		self.sum_spike += t_spike
		return t_spike

	def initialize_state(self, batch_size, device):
		self.memory_voltage = self.V_init.detach().repeat(batch_size, 1).to(device)  # shape: [size, ] -> [batch, size]
		self.memory_spike = torch.zeros(batch_size, self.size).to(device)
		self.sum_spike = torch.zeros(batch_size, self.size).to(device)
		self.ref_th = torch.ones(batch_size, self.size).to(device) * self.t_ref / self.dt
		self.ref = torch.ones(batch_size, self.size).to(device) * self.t_ref / self.dt
		self.device = device

	def reset_state(self):
		self.memory_voltage = None
		self.memory_spike = None
		self.sum_spike = None
		self.ref_th = None
		self.ref = None
		self.device = None