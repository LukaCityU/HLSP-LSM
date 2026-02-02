import torch
import numpy as np
import matplotlib.pyplot as plt
from BaseModel.NeuronModel import LIF


def fun(dt):
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	simulation_time = 100  # ms
	time_step = int(simulation_time / dt)  # time step
	batch_size = 1
	neuron_size = 2

	lif = LIF(neuron_size, np.array([0, ]), V_reset=0., V_th=20, tau_m=10., t_ref=5., R=1., dt=dt)

	c1 = 1.9
	c2 = 2.0
	I = torch.tensor([[c1, c2]]).float().to(device)  # [batch, size]

	V = np.zeros((batch_size, time_step, neuron_size))
	S = np.zeros((batch_size, time_step, neuron_size))
	for t in range(time_step):
		lif(I)
		if device == 'cpu':
			V[:, t, :] = lif.memory_voltage.detach().numpy()
			S[:, t, :] = lif.memory_spike.detach().numpy()
		else:
			V[:, t, :] = lif.memory_voltage.detach().cpu().numpy()
			S[:, t, :] = lif.memory_spike.detach().cpu().numpy()

	voltage1 = V[0, :, 0]
	spike1 = S[0, :, 0:1]

	voltage2 = V[0, :, 1]
	spike2 = S[0, :, 1:2]

	duration = np.arange(time_step) * dt

	_, axs = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[3, 1], sharex='all')
	axs[0].axhline(y=20., ls='--', label='Vth', c='black')
	axs[0].plot(duration, voltage1, label='I = ' + str(c1) + ' nA', c='#1f77b4', lw=2)
	axs[0].plot(duration, voltage2, label='I = ' + str(c2) + ' nA', c='#ff7f0e', lw=2)
	axs[0].legend(edgecolor='black', framealpha=1, loc='lower right')
	axs[0].set(xlabel=r'$t$ (ms)', ylabel=r'$V$ (mV)', title='Minimal Current')
	axs[0].spines['top'].set_visible(False)
	axs[0].spines['right'].set_visible(False)

	elements = np.where(spike1 > 0.)
	index = elements[1] - 0.1
	time = duration[elements[0]]
	axs[1].plot(time, index, '.', markersize=8, c='#1f77b4')
	elements = np.where(spike2 > 0.)
	index = elements[1] + 0.1
	time = duration[elements[0]]
	axs[1].plot(time, index, '.', markersize=8, c='#ff7f0e')
	axs[1].set(xlabel=r'$t$ (ms)', title='Spike', ylabel='', yticks=[], ylim=[-0.2, 0.2])
	axs[1].spines['top'].set_visible(False)
	axs[1].spines['right'].set_visible(False)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	dt_ = 1.0
	fun(dt_)
	print('done')
