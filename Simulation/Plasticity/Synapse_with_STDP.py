import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from BaseModel.PlasticityModel import ESTDP


def fun(dt):
	sim_time = 5
	num_pre = 2
	num_post = 3
	time_steps = int(sim_time / dt)
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

	# s1 = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.]
	# s2 = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]

	# Generate random spike trains
	pre_spikes = torch.randint(0, 2, (1, time_steps, num_pre)).float().to(device)
	post_spikes = torch.randint(0, 2, (1, time_steps, num_post)).float().to(device)

	# Initialize weights
	weights = np.ones((num_post, num_pre)) / 2
	conn = np.where(weights == 0, 0, 1)
	# Remove some connections (random masking)
	conn[1, 0] = 0.
	conn[2, 1] = 0.
	weights = torch.from_numpy(weights).float().to(device)

	# Initialize STDP module
	ESTDP_ = {'lr': 1., 'tau_pre': 10., 'tau_post': 10., 'A1': 0.15, 'A2': 0.15, 'a1': 1., 'a2': 1., 'conn': conn}
	stdp = ESTDP(dt=dt, **ESTDP_)

	# Run STDP update
	dw = torch.zeros(time_steps).float().to(device)
	pre_trace = torch.zeros(time_steps).float().to(device)
	post_trace = torch.zeros(time_steps).float().to(device)
	W = torch.zeros(time_steps).float().to(device)
	# Neuron indices for before/after visualization
	pre_index = 0
	post_index = 0
	batch_index = 0
	for t in range(time_steps):
		dw_t, pre_trace_t, post_trace_t = stdp(pre_spikes[:, t, :], post_spikes[:, t, :], return_more=True)
		weights += dw_t

		dw[t] = dw_t[post_index, pre_index]
		pre_trace[t] = pre_trace_t[batch_index, pre_index]
		post_trace[t] = post_trace_t[batch_index, post_index]
		W[t] = weights[post_index, pre_index]

	dw = dw.to('cpu').numpy()
	W = W.to('cpu').numpy()
	pre_trace = pre_trace.to('cpu').numpy()
	post_trace = post_trace.to('cpu').numpy()
	pre_spikes = pre_spikes.to('cpu').numpy()
	post_spikes = post_spikes.to('cpu').numpy()

	duration = np.arange(time_steps) * dt
	_, axs = plt.subplots(6, 1, figsize=(4, 8), sharex='all')
	axs_list = axs.flatten()

	# pre spike
	elements = np.where(pre_spikes[batch_index, :, pre_index:pre_index + 1] > 0.)
	index = elements[1]
	time = duration[elements[0]]
	axs[0].plot(time, index, '.k', markersize=4)
	axs[0].set(ylabel=r'pre.spike')

	# pre trace
	axs[1].plot(duration, pre_trace)
	axs[1].set(ylabel=r'pre.trace')

	# post spike
	elements = np.where(post_spikes[batch_index, :, post_index:post_index + 1] > 0.)
	index = elements[1]
	time = duration[elements[0]]
	axs[2].plot(time, index, '.k', markersize=4)
	axs[2].set(ylabel=r'post.spike')

	# post trace
	axs[3].plot(duration, post_trace)
	axs[3].set(ylabel=r'post.trace')

	# dw
	axs[4].plot(duration, dw)
	axs[4].set(ylabel=r'dw')

	# w
	axs[5].plot(duration, W)
	axs[5].set(ylabel=r'w', xlabel='t (ms)')

	for ax in axs_list:
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
	plt.tight_layout(pad=0.5, h_pad=1.5)
	plt.show()


if __name__ == '__main__':
	dt_ = 1.
	fun(dt_)

