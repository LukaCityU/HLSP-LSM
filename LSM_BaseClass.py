import torch
import torch.nn as nn

import numpy as np
import random as ran

from BaseModel.NeuronModel import LIF
from BaseModel.SynapseModel import Expon
from BaseModel.PlasticityModel import ESTDP, ISTDP, STD


class WeightGenerator:
	def __init__(self, parameter):
		"""
		Get parameter settings and generate weights.

		:param parameter: dict may contain the following keys:
		    'seed_val'       : int   Random seed for reproducibility
		    'input_size'     : int   Number of input neurons
		    'reservoir_size' : int   Number of reservoir (liquid) neurons
		    'exc_ratio'      : float Proportion of excitatory neurons in the reservoir
		    'w'              : dict  Parameters for initializing reservoir weights W (see specific methods)
		    'w_in'           : dict  Parameters for initializing input weights W_in (see specific methods)

		:return: All initialized weights and masks
		"""
		self.seed_val = parameter['seed_val']
		if self.seed_val is not None:
			np.random.seed(self.seed_val)
			ran.seed(self.seed_val)

		self.N_input = parameter['input_size']  # Input layer size
		self.N_rec = parameter['rec_size']      # Reservoir (liquid) size
		self.N_excitation = int(self.N_rec * parameter['exc_ratio'])  # Number of excitatory neurons
		self.N_inhibition = int(self.N_rec - self.N_excitation)       # Number of inhibitory neurons

		# Main initialization:
		# （1）Liquid layer：
		self.W_M = None  # Reservoir connection mask: [reservoir, reservoir] binary

		# （2）Input layer：
		self.U = None    # Input weight matrix W_in
		self.U_M = None  # Input connection mask

		self.exc_index = None  # Excitatory neuron indices
		self.inh_index = None  # Inhibitory neuron indices

		# Call init function
		self.w_parameters = parameter['w']
		self.init_w()
		self.w_in_parameters = parameter['w_in']
		self.init_w_in()

		self.calculate_conn_num()

	def init_w_in(self):
		"""
		Initialize input layer. Required parameters:
			'input_ratio': float Input-to-reservoir connection ratio
			'value': float Weight value

		:return: Input weight matrix, shape = [reservoir_size, input_size]
		"""
		input_ratio = self.w_in_parameters['input_ratio']
		value = self.w_in_parameters['value']

		# Binary mask: which input-reservoir pairs are connected
		random_mask = np.random.rand(self.N_rec, self.N_input) < input_ratio
		# Randomly sign weights
		sign_matrix = np.where(np.random.rand(self.N_rec, self.N_input) < 0.5, 1, -1)

		self.U = random_mask * sign_matrix * value
		self.U_M = np.where(self.U == 0, 0, 1)

	def init_w(self):
		"""
		Initialize reservoir/liquid connections. Required parameters:
			'k_prob': dict Connection probabilities for each synapse type {'EE', 'EI', 'IE', 'II'}

		:return: Connection mask
		"""
		k_prob = self.w_parameters['k_prob']
		N_rec = self.N_rec
		N_excitation = self.N_excitation

		# Set top 80% to 1, bottom 20% to 0 (for structured masking)
		neuron_type = [int(i < N_excitation) for i in range(N_rec)]
		neuron_type = np.array(neuron_type)
		exc_index = np.where(neuron_type == 1)[0]
		inh_index = np.where(neuron_type == 0)[0]
		self.exc_index = exc_index
		self.inh_index = inh_index

		C = np.zeros((N_rec, N_rec))  # Four probability matrices [out, in] for EE/EI/IE/II
		C[: N_excitation, : N_excitation] = k_prob['EE']
		C[N_excitation: , : N_excitation] = k_prob['EI']
		C[: N_excitation, N_excitation: ] = k_prob['IE']
		C[N_excitation: , N_excitation: ] = k_prob['II']
		np.fill_diagonal(C, 0)  # Remove self-connections (diagonal = 0)

		# Sample connections from probabilities
		W_conn = np.random.random((N_rec, N_rec))
		W_conn = np.where(C < W_conn, 0, 1)
		self.W_M = W_conn

	def calculate_conn_num(self):
		"""
        Count actual synapse numbers per type: [EE, EI, IE, II]
        :return:
        """
		w_final = self.W_M

		temp = w_final[:, self.exc_index]  # shape = [out, in]
		temp = temp[self.exc_index, :]
		c1 = len(np.argwhere(temp != 0))

		temp = w_final[:, self.exc_index]
		temp = temp[self.inh_index, :]
		c2 = len(np.argwhere(temp != 0))

		temp = w_final[:, self.inh_index]
		temp = temp[self.exc_index, :]
		c3 = len(np.argwhere(temp != 0))

		temp = w_final[:, self.inh_index]
		temp = temp[self.inh_index, :]
		c4 = len(np.argwhere(temp != 0))

		print("Total synapse:", len(np.argwhere(w_final != 0)),
			  ",E --> E:", c1,
			  ",E --> I:", c2,
			  ",I --> E:", c3,
			  ",I --> I:", c4)
		return c1, c2, c3, c4


class LSMParameters:
	def __init__(self, input_size, rec_size, dt, seed=None, w_in=0.25, beta_E=1., beta_I=1.):
		input_size = input_size
		rec_size = rec_size
		exc_ratio = 0.8

		# region （1）Generate matrix
		w_in = {'input_ratio': 0.15, 'value': w_in}
		w = {'k_prob': {'EE': 0.1, 'EI': 0.1, 'IE': 0.1, 'II': 0.1}}
		generator_parameter = {'input_size': input_size, 'rec_size': rec_size, 'exc_ratio': exc_ratio,
							   'w': w, 'w_in': w_in, 'seed_val': seed}
		wg = WeightGenerator(generator_parameter)

		w_c = wg.W_M  # Connection mask (1 = connected, 0 = no synapse) shape = [out, in]
		N_excitation = wg.N_excitation

		ee = np.zeros((wg.N_excitation, wg.N_excitation))  # Weight matrix (start with zeros)
		ei = np.zeros((wg.N_inhibition, wg.N_excitation))
		ie = np.zeros((wg.N_excitation, wg.N_inhibition))
		ii = np.zeros((wg.N_inhibition, wg.N_inhibition))

		ee_c = w_c[: N_excitation, : N_excitation]  # E -> E
		ei_c = w_c[N_excitation: , : N_excitation]  # E -> I
		ie_c = w_c[: N_excitation, N_excitation: ]  # I -> E
		ii_c = w_c[N_excitation: , N_excitation: ]  # I -> I

		u = wg.U
		# Input -> E
		ine = u[: N_excitation, :]
		ine_pos = np.where(ine > 0, ine, 0)
		ine_pos_c = np.where(ine > 0, 1, 0)
		ine_neg = np.where(ine < 0, ine, 0)
		# Force inhibitory weights to be positive; inhibitory effect comes from E_rev = -1 in neuron dynamics
		ine_neg = np.abs(ine_neg)
		ine_neg_c = np.where(ine < 0, 1, 0)
		# Input -> I
		ini = u[N_excitation: , :]
		ini_pos = np.where(ini > 0, ini, 0)
		ini_pos_c = np.where(ini > 0, 1, 0)
		ini_neg = np.where(ini < 0, ini, 0)
		ini_neg = np.abs(ini_neg)
		ini_neg_c = np.where(ini < 0, 1, 0)
		# endregion

		# region （2）Network parameters
		# Neuron parameters: V_init, V_reset, V_th, tau_m, t_ref
		temp1 = np.zeros((ee.shape[0],))
		E_neuron = {'V_init': temp1, 'V_reset': 0., 'V_th': 20., 'tau_m': 64., 't_ref': 2.}
		temp2 = np.zeros((ii.shape[0],))
		I_neuron = {'V_init': temp2, 'V_reset': 0., 'V_th': 20., 'tau_m': 64., 't_ref': 2.}
		# Synapse parameters: weight, tau_syn E, dt
		inpE_pos = {'weight': ine_pos, 'conn': ine_pos_c, 'tau_syn': 1., 'E':  1.}
		inpE_neg = {'weight': ine_neg, 'conn': ine_neg_c, 'tau_syn': 1., 'E': -1.}
		inpI_pos = {'weight': ini_pos, 'conn': ini_pos_c, 'tau_syn': 1., 'E':  1.}
		inpI_neg = {'weight': ini_neg, 'conn': ini_neg_c, 'tau_syn': 1., 'E': -1.}
		EE = {'weight': ee, 'conn': ee_c, 'tau_syn': 1., 'E':  1.}
		EI = {'weight': ei, 'conn': ei_c, 'tau_syn': 1., 'E':  1.}
		IE = {'weight': ie, 'conn': ie_c, 'tau_syn': 1., 'E': -1.}
		II = {'weight': ii, 'conn': ii_c, 'tau_syn': 1., 'E': -1.}
		# ESTDP parameters: lr, tau_pre, tau_post, A1, A2, dt
		ESTDP_ = {'lr': 1., 'tau_pre': 10., 'tau_post': 10., 'A1': 0.15, 'A2': 0.15 * beta_E, 'a1': 0.1, 'a2': 0.1}
		ESTDP2_ = {'lr': 1., 'tau_pre': 10., 'tau_post': 10., 'A1': 0.15, 'A2': 0.15 * beta_I, 'a1': 0.1, 'a2': 0.1}
		# ISTDP parameters: lr, tau1, tau2, A, beta, dt
		ISTDP_ = {'lr': 1., 'tau1': 5., 'tau2': 10., 'A': 0.15, 'beta': beta_I, 'a': 0.1}
		# STP parameters: in_dim, U, tau_d, dt
		STP_ = {'U': 0.4, 'tau_d': 150.}
		self.parameter = {'input_size': input_size, 'rec_size': rec_size, 'E_ratio': exc_ratio, 'dt': dt, 'W_max': 3.,
						  'E_neuron': E_neuron, 'I_neuron': I_neuron,
						  'inpE_pos': inpE_pos, 'inpE_neg': inpE_neg, 'inpI_pos': inpI_pos, 'inpI_neg': inpI_neg,
						  'EE': EE, 'EI': EI, 'IE': IE, 'II': II,
						  'ESTDP': ESTDP_, 'ESTDP2': ESTDP2_, 'ISTDP': ISTDP_, 'STP': STP_}
		# endregion
		print('Parameters have been created')


class LSM(nn.Module):
	def __init__(self, parameter):
		super().__init__()
		# Initial parameters
		self.input_size = parameter['input_size']
		self.rec_size = int(np.prod(parameter['rec_size']))
		self.E_ratio = parameter['E_ratio']
		self.dt = parameter['dt']
		self.E_size = int(self.rec_size * self.E_ratio)
		self.I_size = int(self.rec_size - self.E_size)
		self.init_flag = False

		# Neuron group
		self.E = LIF(size=self.E_size, **parameter['E_neuron'])
		self.I = LIF(size=self.I_size, **parameter['I_neuron'])

		# Synapse
		self.inpE_pos = Expon(dt=self.dt, **parameter['inpE_pos'])  # External excitatory input current to E population
		self.inpE_neg = Expon(dt=self.dt, **parameter['inpE_neg'])  # External inhibitory input current to E population
		self.inpI_pos = Expon(dt=self.dt, **parameter['inpI_pos'])
		self.inpI_neg = Expon(dt=self.dt, **parameter['inpI_neg'])
		self.EE = Expon(dt=self.dt, **parameter['EE'])
		self.EI = Expon(dt=self.dt, **parameter['EI'])
		self.IE = Expon(dt=self.dt, **parameter['IE'])
		self.II = Expon(dt=self.dt, **parameter['II'])

	def forward(self, x, record=False, **kwargs):
		"""
		:param x: Input data, shape [batch, time, input_size]
		:param record: If True, store membrane voltages and synaptic currents
		:return:
		"""
		raise NotImplementedError(f'Subclass of {self.__class__.__name__} must implement "forward" function.')

	def initialize_state(self, batch_size, device):
		self.E.initialize_state(batch_size, device)
		self.I.initialize_state(batch_size, device)
		self.inpE_pos.initialize_state(batch_size, device)
		self.inpE_neg.initialize_state(batch_size, device)
		self.inpI_pos.initialize_state(batch_size, device)
		self.inpI_neg.initialize_state(batch_size, device)
		self.EE.initialize_state(batch_size, device)
		self.EI.initialize_state(batch_size, device)
		self.IE.initialize_state(batch_size, device)
		self.II.initialize_state(batch_size, device)
		self.init_flag = True

	def reset_state(self):
		self.E.reset_state()
		self.I.reset_state()
		self.inpE_pos.reset_state()
		self.inpE_neg.reset_state()
		self.inpI_pos.reset_state()
		self.inpI_neg.reset_state()
		self.EE.reset_state()
		self.EI.reset_state()
		self.IE.reset_state()
		self.II.reset_state()
		self.init_flag = False


class LSMESTDP(LSM):
	def __init__(self, parameter):
		super().__init__(parameter=parameter)
		self.W_max = float(parameter['W_max'])

		self.ESTDPLearnerEE = ESTDP(conn=parameter['EE']['conn'], dt=self.dt, **parameter['ESTDP'])
		self.ESTDPLearnerEI = ESTDP(conn=parameter['EI']['conn'], dt=self.dt, **parameter['ESTDP'])
		self.ESTDPLearnerIE = ESTDP(conn=parameter['IE']['conn'], dt=self.dt, **parameter['ESTDP2'])
		self.ESTDPLearnerII = ESTDP(conn=parameter['II']['conn'], dt=self.dt, **parameter['ESTDP2'])

	def forward(self, x, record=False, **kwargs):
		if record:
			NGR = NeuronGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
			SGR = SynapseGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
		else:
			NGR = None
			SGR = None

		if not self.init_flag:
			self.initialize_state(x.shape[0], x.device)
		sum_spike_E = torch.zeros(x.shape[0], self.E_size, device=x.device)
		sum_spike_I = torch.zeros(x.shape[0], self.I_size, device=x.device)

		if x.ndim == 2:    # Rate-coded data (MNIST etc.) → firing probability per time step
			inp_sps = torch.zeros(x.shape[0], kwargs['time_step'], x.shape[1])  # [batch, time, input_size]
		elif x.ndim == 3:  # Already spiking data
			inp_sps = x
		else:
			raise ValueError("Input type error.")

		for time_step in range(kwargs['time_step']):
			# (1) Sample input spikes at current timestep
			if x.ndim == 2:
				random_tensor = torch.rand_like(x)  # Uniform random matrix for Bernoulli sampling
				input_spike = torch.lt(random_tensor, x).float()  # shape = [batch, input_size]
				inp_sps[:, time_step, :] = input_spike
			else:
				input_spike = x[:, time_step, :]

			# (2) Input current to E population → [batch, E_size]
			a_E_pos = self.inpE_pos(input_spike)
			a_E_neg = self.inpE_neg(input_spike)
			b_E = self.EE(self.E.memory_spike)
			c_E = self.IE(self.I.memory_spike)
			current_to_E = a_E_pos + a_E_neg + b_E + c_E

			# (3) Input current to I population → [batch, I_size]
			a_I_pos = self.inpI_pos(input_spike)
			a_I_neg = self.inpI_neg(input_spike)
			b_I = self.EI(self.E.memory_spike)
			c_I = self.II(self.I.memory_spike)
			current_to_I = a_I_pos + a_I_neg + b_I + c_I

			# (4) Get presynaptic spikes from t-1
			pre_spike_E = self.E.memory_spike.clone()
			pre_spike_I = self.I.memory_spike.clone()

			# (5) Generate E and I output spikes, shape = [batch, E/I_size]
			s_E = self.E(current_to_E)
			s_I = self.I(current_to_I)

			# (6) STDP update on liquid weights W
			if kwargs['STDP']:
				# ESTDP
				self.EE.W += self.ESTDPLearnerEE(pre_spike_E, s_E)
				self.EI.W += self.ESTDPLearnerEI(pre_spike_E, s_I)
				self.IE.W += self.ESTDPLearnerIE(pre_spike_I, s_E)
				self.II.W += self.ESTDPLearnerII(pre_spike_I, s_I)
				# Clip weights to bounds
				self.clip_weight()

			# (7) Accumulate total spike count (for monitoring activity)
			sum_spike_E += s_E
			sum_spike_I += s_I

			# (8) Record internal variables if needed
			if record:
				NGR.update(E=self.E, I=self.I, time_index=time_step)
				SGR.update(time_index=time_step,
						   inpE_pos=self.inpE_pos, inpE_neg=self.inpE_neg, EE=self.EE, IE=self.IE,
						   inpI_pos=self.inpI_pos, inpI_neg=self.inpI_neg, EI=self.EI, II=self.II)

		# (9) Reset neuronal/synaptic states between trials
		self.reset_state()
		if record:
			return inp_sps, sum_spike_E, sum_spike_I, NGR, SGR
		else:
			return inp_sps, sum_spike_E, sum_spike_I

	def initialize_state(self, batch_size, device):
		super().initialize_state(batch_size, device)
		self.ESTDPLearnerEE.initialize_traces(batch_size, self.E_size, self.E_size, device)
		self.ESTDPLearnerEI.initialize_traces(batch_size, self.E_size, self.I_size, device)
		self.ESTDPLearnerIE.initialize_traces(batch_size, self.I_size, self.E_size, device)
		self.ESTDPLearnerII.initialize_traces(batch_size, self.I_size, self.I_size, device)

	def reset_state(self):
		super().reset_state()
		self.ESTDPLearnerEE.reset_traces()
		self.ESTDPLearnerEI.reset_traces()
		self.ESTDPLearnerIE.reset_traces()
		self.ESTDPLearnerII.reset_traces()

	def clip_weight(self):
		self.EE.W.clip_(min=0, max=self.W_max)
		self.EI.W.clip_(min=0, max=self.W_max)
		self.IE.W.clip_(min=0, max=self.W_max)
		self.II.W.clip_(min=0, max=self.W_max)


class LSMEISTDP(LSM):
	def __init__(self, parameter):
		super().__init__(parameter=parameter)
		self.W_max = float(parameter['W_max'])

		self.ESTDPLearnerEE = ESTDP(conn=parameter['EE']['conn'], dt=self.dt, **parameter['ESTDP'])
		self.ESTDPLearnerEI = ESTDP(conn=parameter['EI']['conn'], dt=self.dt, **parameter['ESTDP'])
		self.ISTDPLearnerIE = ISTDP(conn=parameter['IE']['conn'], dt=self.dt, **parameter['ISTDP'])
		self.ISTDPLearnerII = ISTDP(conn=parameter['II']['conn'], dt=self.dt, **parameter['ISTDP'])

	def forward(self, x, record=False, **kwargs):
		if record:
			NGR = NeuronGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
			SGR = SynapseGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
		else:
			NGR = None
			SGR = None

		if not self.init_flag:
			self.initialize_state(x.shape[0], x.device)

		sum_spike_E = torch.zeros(x.shape[0], self.E_size, device=x.device)
		sum_spike_I = torch.zeros(x.shape[0], self.I_size, device=x.device)
		if x.ndim == 2:
			inp_sps = torch.zeros(x.shape[0], kwargs['time_step'], x.shape[1])  # [batch, time, input_size]
		elif x.ndim == 3:
			inp_sps = x
		else:
			raise ValueError("Input type error.")

		for time_step in range(kwargs['time_step']):
			# (1) Sample input spikes at current timestep
			if x.ndim == 2:
				random_tensor = torch.rand_like(x)
				input_spike = torch.lt(random_tensor, x).float()
				inp_sps[:, time_step, :] = input_spike
			else:
				input_spike = x[:, time_step, :]

			# (2) Input current to E population → [batch, E_size]
			a_E_pos = self.inpE_pos(input_spike)
			a_E_neg = self.inpE_neg(input_spike)
			b_E = self.EE(self.E.memory_spike)
			c_E = self.IE(self.I.memory_spike)
			current_to_E = a_E_pos + a_E_neg + b_E + c_E

			# (3) Input current to I population → [batch, I_size]
			a_I_pos = self.inpI_pos(input_spike)
			a_I_neg = self.inpI_neg(input_spike)
			b_I = self.EI(self.E.memory_spike)
			c_I = self.II(self.I.memory_spike)
			current_to_I = a_I_pos + a_I_neg + b_I + c_I

			# (4) Get presynaptic spikes from t-1
			pre_spike_E = self.E.memory_spike.clone()
			pre_spike_I = self.I.memory_spike.clone()

			# (5) Generate E and I output spikes
			s_E = self.E(current_to_E)
			s_I = self.I(current_to_I)

			# (6) STDP update on recurrent weights W
			if kwargs['STDP']:
				# ESTDP
				self.EE.W += self.ESTDPLearnerEE(pre_spike_E, s_E)
				self.EI.W += self.ESTDPLearnerEI(pre_spike_E, s_I)
				# ISTDP
				self.IE.W += self.ISTDPLearnerIE(pre_spike_I, s_E)
				self.II.W += self.ISTDPLearnerII(pre_spike_I, s_I)
				# Clip weights to bounds
				self.clip_weight()

			# (7) Accumulate total spike count (for monitoring activity)
			sum_spike_E += s_E
			sum_spike_I += s_I

			# (8) Record internal variables if needed
			if record:
				NGR.update(E=self.E, I=self.I, time_index=time_step)
				SGR.update(time_index=time_step,
						   inpE_pos=self.inpE_pos, inpE_neg=self.inpE_neg, EE=self.EE, IE=self.IE,
						   inpI_pos=self.inpI_pos, inpI_neg=self.inpI_neg, EI=self.EI, II=self.II)

		# (9) Reset neuronal/synaptic states between trials
		self.reset_state()
		if record:
			return inp_sps, sum_spike_E, sum_spike_I, NGR, SGR
		else:
			return inp_sps, sum_spike_E, sum_spike_I

	def initialize_state(self, batch_size, device):
		super().initialize_state(batch_size, device)
		self.ESTDPLearnerEE.initialize_traces(batch_size, self.E_size, self.E_size, device)
		self.ESTDPLearnerEI.initialize_traces(batch_size, self.E_size, self.I_size, device)
		self.ISTDPLearnerIE.initialize_traces(batch_size, self.I_size, self.E_size, device)
		self.ISTDPLearnerII.initialize_traces(batch_size, self.I_size, self.I_size, device)

	def reset_state(self):
		super().reset_state()
		self.ESTDPLearnerEE.reset_traces()
		self.ESTDPLearnerEI.reset_traces()
		self.ISTDPLearnerIE.reset_traces()
		self.ISTDPLearnerII.reset_traces()

	def clip_weight(self):
		self.EE.W.clip_(min=0, max=self.W_max)
		self.EI.W.clip_(min=0, max=self.W_max)
		self.IE.W.clip_(min=0, max=self.W_max)
		self.II.W.clip_(min=0, max=self.W_max)


class LSMESTDPSTP(LSMESTDP):
	def __init__(self, parameter):
		super().__init__(parameter=parameter)
		self.STD_E = STD(in_dim=self.E_size, dt=self.dt, **parameter['STP'])
		self.STD_I = STD(in_dim=self.I_size, dt=self.dt, **parameter['STP'])

	def forward(self, x, record=False, **kwargs):
		if record:
			NGR = NeuronGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
			SGR = SynapseGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
		else:
			NGR = None
			SGR = None

		if not self.init_flag:
			self.initialize_state(x.shape[0], x.device)

		sum_spike_E = torch.zeros(x.shape[0], self.E_size, device=x.device)
		sum_spike_I = torch.zeros(x.shape[0], self.I_size, device=x.device)
		if x.ndim == 2:
			inp_sps = torch.zeros(x.shape[0], kwargs['time_step'], x.shape[1])  # [batch, time, input_size]
		elif x.ndim == 3:
			inp_sps = x
		else:
			raise ValueError("Input type error.")

		for time_step in range(kwargs['time_step']):
			# (1) Sample input spikes at current timestep
			if x.ndim == 2:
				random_tensor = torch.rand_like(x)
				input_spike = torch.lt(random_tensor, x).float()  # shape = [batch, input_size]
				inp_sps[:, time_step, :] = input_spike
			else:
				input_spike = x[:, time_step, :]

			# (2) Get presynaptic spikes from t-1
			pre_spike_E = self.E.memory_spike.clone()
			pre_spike_I = self.I.memory_spike.clone()

			# (3) Compute recurrent synaptic currents (including short-term depression dynamics)
			pre_current_E = self.STD_E(pre_spike_E)
			pre_current_I = self.STD_I(pre_spike_I)

			# （4）Input current to E population → [batch, E_size]
			a_E_pos = self.inpE_pos(input_spike)
			a_E_neg = self.inpE_neg(input_spike)
			b_E = self.EE(pre_current_E)
			c_E = self.IE(pre_current_I)
			current_to_E = a_E_pos + a_E_neg + b_E + c_E

			# （5）Input current to I population → [batch, I_size]
			a_I_pos = self.inpI_pos(input_spike)
			a_I_neg = self.inpI_neg(input_spike)
			b_I = self.EI(pre_current_E)
			c_I = self.II(pre_current_I)
			current_to_I = a_I_pos + a_I_neg + b_I + c_I

			# （6）Generate E and I output spikes
			s_E = self.E(current_to_E)
			s_I = self.I(current_to_I)

			# （7）STDP update on recurrent weights W
			if kwargs['STDP']:
				# ESTDP
				self.EE.W += self.ESTDPLearnerEE(pre_spike_E, s_E)
				self.EI.W += self.ESTDPLearnerEI(pre_spike_E, s_I)
				self.IE.W += self.ESTDPLearnerIE(pre_spike_I, s_E)
				self.II.W += self.ESTDPLearnerII(pre_spike_I, s_I)
				# Clip weights to bounds
				self.clip_weight()

			# （8）Accumulate total spike count (for monitoring activity)
			sum_spike_E += s_E
			sum_spike_I += s_I

			# （9）Record internal variables if needed
			if record:
				NGR.update(E=self.E, I=self.I, time_index=time_step)
				SGR.update(time_index=time_step,
						   inpE_pos=self.inpE_pos, inpE_neg=self.inpE_neg, EE=self.EE, IE=self.IE,
						   inpI_pos=self.inpI_pos, inpI_neg=self.inpI_neg, EI=self.EI, II=self.II)

		# Reset neuronal/synaptic states between trials
		self.reset_state()
		if record:
			return inp_sps, sum_spike_E, sum_spike_I, NGR, SGR
		else:
			return inp_sps, sum_spike_E, sum_spike_I

	def initialize_state(self, batch_size, device):
		super().initialize_state(batch_size, device)
		self.STD_E.initialize_state(batch_size, device)
		self.STD_I.initialize_state(batch_size, device)

	def reset_state(self):
		super().reset_state()
		self.STD_E.reset_state()
		self.STD_I.reset_state()

	def clip_weight(self):
		super().clip_weight()


class LSMEISTDPSTP(LSMEISTDP):
	def __init__(self, parameter):
		super().__init__(parameter=parameter)
		self.STD_E = STD(in_dim=self.E_size, dt=self.dt, **parameter['STP'])
		self.STD_I = STD(in_dim=self.I_size, dt=self.dt, **parameter['STP'])

	def forward(self, x, record=False, **kwargs):
		if record:
			NGR = NeuronGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
			SGR = SynapseGroupRecord(x.shape[0], self.E_size, self.I_size, kwargs['time_step'])
		else:
			NGR = None
			SGR = None

		if not self.init_flag:
			self.initialize_state(x.shape[0], x.device)

		sum_spike_E = torch.zeros(x.shape[0], self.E_size, device=x.device)
		sum_spike_I = torch.zeros(x.shape[0], self.I_size, device=x.device)
		if x.ndim == 2:
			inp_sps = torch.zeros(x.shape[0], kwargs['time_step'], x.shape[1])  # [batch, time, input_size]
		elif x.ndim == 3:
			inp_sps = x
		else:
			raise ValueError("Input type error.")

		for time_step in range(kwargs['time_step']):
			# （1）Sample input spikes at current timestep
			if x.ndim == 2:
				random_tensor = torch.rand_like(x)
				input_spike = torch.lt(random_tensor, x).float()
				inp_sps[:, time_step, :] = input_spike
			else:
				input_spike = x[:, time_step, :]

			# （2）Get presynaptic spikes from t-1
			pre_spike_E = self.E.memory_spike.clone()
			pre_spike_I = self.I.memory_spike.clone()

			# （3）Compute recurrent synaptic currents (including short-term depression dynamics)
			pre_current_E = self.STD_E(pre_spike_E)
			pre_current_I = self.STD_I(pre_spike_I)

			# （4）Input current to E population → [batch, E_size]
			a_E_pos = self.inpE_pos(input_spike)
			a_E_neg = self.inpE_neg(input_spike)
			b_E = self.EE(pre_current_E)
			c_E = self.IE(pre_current_I)
			current_to_E = a_E_pos + a_E_neg + b_E + c_E

			# （5）Input current to I population → [batch, I_size]
			a_I_pos = self.inpI_pos(input_spike)
			a_I_neg = self.inpI_neg(input_spike)
			b_I = self.EI(pre_current_E)
			c_I = self.II(pre_current_I)
			current_to_I = a_I_pos + a_I_neg + b_I + c_I

			# （6）Generate E and I output spikes
			s_E = self.E(current_to_E)
			s_I = self.I(current_to_I)

			# （7）STDP update on recurrent weights W
			if kwargs['STDP']:
				# ESTDP
				self.EE.W += self.ESTDPLearnerEE(pre_spike_E, s_E)
				self.EI.W += self.ESTDPLearnerEI(pre_spike_E, s_I)
				# ISTDP
				self.IE.W += self.ISTDPLearnerIE(pre_spike_I, s_E)
				self.II.W += self.ISTDPLearnerII(pre_spike_I, s_I)
				# Clip weights to bounds
				self.clip_weight()

			# （8）Accumulate total spike count (for monitoring activity)
			sum_spike_E += s_E
			sum_spike_I += s_I

			# （9）Record internal variables if needed
			if record:
				NGR.update(E=self.E, I=self.I, time_index=time_step)
				SGR.update(time_index=time_step,
				           inpE_pos=self.inpE_pos, inpE_neg=self.inpE_neg, EE=self.EE, IE=self.IE,
				           inpI_pos=self.inpI_pos, inpI_neg=self.inpI_neg, EI=self.EI, II=self.II)

		# Reset neuronal/synaptic states between trials
		self.reset_state()
		if record:
			return inp_sps, sum_spike_E, sum_spike_I, NGR, SGR
		else:
			return inp_sps, sum_spike_E, sum_spike_I

	def initialize_state(self, batch_size, device):
		super().initialize_state(batch_size, device)
		self.STD_E.initialize_state(batch_size, device)
		self.STD_I.initialize_state(batch_size, device)

	def reset_state(self):
		super().reset_state()
		self.STD_E.reset_state()
		self.STD_I.reset_state()

	def clip_weight(self):
		super().clip_weight()


class NeuronGroupRecord:
	def __init__(self, batch_size: int, E_size: int, I_size: int, time_step: int):
		if E_size > 0:
			self.S_E = np.zeros((batch_size, time_step, E_size))

		if I_size > 0:
			self.S_I = np.zeros((batch_size, time_step, I_size))

	def update(self, E=None, I=None, time_index: int=None):
		if E is not None:
			if E.device == 'cpu':
				self.S_E[:, time_index, :] = E.memory_spike.detach().numpy()
			else:
				self.S_E[:, time_index, :] = E.memory_spike.detach().cpu().numpy()

		if I is not None:
			if I.device == 'cpu':
				self.S_I[:, time_index, :] = I.memory_spike.detach().numpy()
			else:
				self.S_I[:, time_index, :] = I.memory_spike.detach().cpu().numpy()


class SynapseGroupRecord:
	def __init__(self, batch_size: int, E_size: int, I_size: int, time_step: int):
		if E_size > 0:
			self.inpE_pos = np.zeros((batch_size, time_step, E_size))
			self.inpE_neg = np.zeros((batch_size, time_step, E_size))
			self.EE = np.zeros((batch_size, time_step, E_size))
			self.IE = np.zeros((batch_size, time_step, E_size))

		if I_size > 0:
			self.inpI_pos = np.zeros((batch_size, time_step, I_size))
			self.inpI_neg = np.zeros((batch_size, time_step, I_size))
			self.EI = np.zeros((batch_size, time_step, I_size))
			self.II = np.zeros((batch_size, time_step, I_size))

	def update(self, time_index: int = None, inpE_pos=None, inpE_neg=None, EE=None, IE=None,
			   inpI_pos=None, inpI_neg=None, EI=None, II=None):
		if inpE_pos is not None:
			if inpE_pos.device == 'cpu':
				self.inpE_pos[:, time_index, :] = inpE_pos.memory_I.detach().numpy()
			else:
				self.inpE_pos[:, time_index, :] = inpE_pos.memory_I.detach().cpu().numpy()

		if inpE_neg is not None:
			if inpE_neg.device == 'cpu':
				self.inpE_neg[:, time_index, :] = inpE_neg.memory_I.detach().numpy()
			else:
				self.inpE_neg[:, time_index, :] = inpE_neg.memory_I.detach().cpu().numpy()

		if EE is not None:
			if EE.device == 'cpu':
				self.EE[:, time_index, :] = EE.memory_I.detach().numpy()
			else:
				self.EE[:, time_index, :] = EE.memory_I.detach().cpu().numpy()

		if IE is not None:
			if IE.device == 'cpu':
				self.IE[:, time_index, :] = IE.memory_I.detach().numpy()
			else:
				self.IE[:, time_index, :] = IE.memory_I.detach().cpu().numpy()

		if inpI_pos is not None:
			if inpI_pos.device == 'cpu':
				self.inpI_pos[:, time_index, :] = inpI_pos.memory_I.detach().numpy()
			else:
				self.inpI_pos[:, time_index, :] = inpI_pos.memory_I.detach().cpu().numpy()

		if inpI_neg is not None:
			if inpI_neg.device == 'cpu':
				self.inpI_neg[:, time_index, :] = inpI_neg.memory_I.detach().numpy()
			else:
				self.inpI_neg[:, time_index, :] = inpI_neg.memory_I.detach().cpu().numpy()

		if EI is not None:
			if EI.device == 'cpu':
				self.EI[:, time_index, :] = EI.memory_I.detach().numpy()
			else:
				self.EI[:, time_index, :] = EI.memory_I.detach().cpu().numpy()

		if II is not None:
			if II.device == 'cpu':
				self.II[:, time_index, :] = II.memory_I.detach().numpy()
			else:
				self.II[:, time_index, :] = II.memory_I.detach().cpu().numpy()

