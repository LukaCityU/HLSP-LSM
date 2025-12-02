import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from BaseModel.SynapseModel import Expon
from BaseModel.NeuronModel import LIF


def fun(dt):
	input_size = 1
	simulation_time = 100  # ms
	time_step = int(simulation_time / dt)  # time step
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

	# Generate input: [batch, time, neuron]
	temp = [int(i / dt) for i in [10, 30, 50, 60, 70, 80]]
	spike_time = np.array(temp)
	input_spike = np.zeros((1, time_step, input_size))
	input_spike[:, spike_time, :] = 1

	# Simulate E synapses
	w_ee = np.ones((1, 1))
	conn = np.ones((1, 1))
	synapse_expon_E = Expon(w_ee, conn, tau_syn=1., E=1., dt=dt).to(device)
	post_neuron_1 = LIF(1, np.array([0, ]), V_reset=0., V_th=20, tau_m=10., t_ref=5., R=1., dt=dt).to(device)

	# Simulate I synapses
	w_ii = np.ones((1, 1))
	synapse_expon_I = Expon(w_ii, conn, tau_syn=1., E=-1., dt=dt).to(device)
	post_neuron_2 = LIF(1, np.array([0, ]), V_reset=0., V_th=20, tau_m=10., t_ref=5., R=1., dt=dt).to(device)

	# Record - Neuron
	V = np.zeros((input_size, time_step, 2))
	# Record - Synapse
	C = np.zeros((input_size, time_step, 2))

	# Run simulation
	synapse_expon_E.eval()
	synapse_expon_I.eval()
	post_neuron_1.eval()
	post_neuron_2.eval()
	with torch.no_grad():
		for t in range(time_step):
			s_pre = torch.tensor(input_spike[:, t, :]).float().to(device)
			E_t = synapse_expon_E(s_pre)  # Postsynaptic current
			I_t = synapse_expon_I(s_pre)
			post_neuron_1(E_t)
			post_neuron_2(I_t)
			if device == 'cpu':
				V[:, t, :1] = post_neuron_1.memory_voltage.detach().numpy()
				V[:, t, 1:] = post_neuron_2.memory_voltage.detach().numpy()
				C[:, t, :1] = synapse_expon_E.memory_I.detach().numpy()
				C[:, t, 1:] = synapse_expon_I.memory_I.detach().numpy()
			else:
				V[:, t, :1] = post_neuron_1.memory_voltage.detach().cpu().numpy()
				V[:, t, 1:] = post_neuron_2.memory_voltage.detach().cpu().numpy()
				C[:, t, :1] = synapse_expon_E.memory_I.detach().cpu().numpy()
				C[:, t, 1:] = synapse_expon_I.memory_I.detach().cpu().numpy()

	# Plot results
	input_spike = input_spike[0]
	duration = np.arange(time_step) * dt
	_, axs = plt.subplots(3, 2, sharex='all')
	axs_list = axs.flatten()

	# E 突触输入
	elements = np.where(input_spike[:, 0:1] > 0.)
	index = elements[1]
	time = duration[elements[0]]
	axs[0, 0].plot(time, index, '.k', markersize=4)
	axs[0, 0].set(ylabel=r'pre.spike', title='E synapse')
	PSC = C[0, :, 0]
	axs[1, 0].plot(duration, PSC)
	axs[1, 0].set(ylabel=r'EPSC')
	V1 = V[0, :, 0]
	axs[2, 0].plot(duration, V1)
	axs[2, 0].set(ylabel=r'post.V', xlabel='t (ms)')
	print('(EE) max PSC:', np.max(PSC), ' max post.V:', np.max(V))

	# I 突触输入
	axs[0, 1].plot(time, index, '.k', markersize=4)
	axs[0, 1].set(ylabel=r'pre.spike', title='I synapse')
	PSC = C[0, :, 1]
	axs[1, 1].plot(duration, PSC)
	axs[1, 1].set(ylabel=r'IPSC')
	V2 = V[0, :, 1]
	axs[2, 1].plot(duration, V2)
	axs[2, 1].set(ylabel=r'post.V', xlabel='t (ms)')
	print('(II) min PSC:', np.min(PSC), ' min post.V:', np.min(V))

	for ax in axs_list:
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
	plt.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.5)
	plt.show()


if __name__ == '__main__':
	dt_ = 0.1
	fun(dt_)
	print('done')