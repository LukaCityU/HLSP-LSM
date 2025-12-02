import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from BaseModel.PlasticityModel import ISTDP, ESTDP
import time


# noinspection PyTypeChecker
def fun1(dt):
	start = time.time()
	time_ = 50
	time_steps = int(time_ / dt)
	spike_train = np.eye(time_steps)
	# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'

	conn = np.ones((1, 1))
	ISTDP_ = {'lr': 1., 'tau1': 5., 'tau2': 10., 'A': 0.15, 'beta': 1.2, 'a': 1., 'conn':conn}
	stdp = ISTDP(dt=dt, **ISTDP_).to(device)

	dt_list = []
	dw_list = []

	post_spikes = torch.zeros((1, time_steps, 1)).to(device)
	post_spikes[0, :, 0] = torch.tensor(spike_train[0])
	t_post = np.argmax(spike_train[0])
	for i in range(time_steps):
		t_pre = np.argmax(spike_train[-i])
		dt_list.append((t_post - t_pre) * dt)
		pre_spikes = torch.zeros((1, time_steps, 1)).to(device)
		pre_spikes[0, :, 0] = torch.tensor(spike_train[-i])
		dw_final = 0.
		for j in range(time_steps):
			pre_spike_t = pre_spikes[:, j, :]
			post_spike_t = post_spikes[:, j, :]
			dw = stdp(pre_spike_t, post_spike_t)
			dw = dw.to('cpu').numpy()
			if dw[0, 0] != 0.:
				dw_final = dw[0, 0]
		dw_list.append(dw_final)
		stdp.reset_traces()

	pre_spikes = torch.zeros((1, time_steps, 1)).to(device)
	pre_spikes[0, :, 0] = torch.tensor(spike_train[0])
	t_pre = np.argmax(spike_train[0])
	for i in range(time_steps):
		t_post = np.argmax(spike_train[i])
		dt_list.append((t_post - t_pre) * dt)
		post_spikes = torch.zeros((1, time_steps, 1)).to(device)
		post_spikes[0, :, 0] = torch.tensor(spike_train[i])
		dw_final = 0.
		for j in range(time_steps):
			pre_spike_t = pre_spikes[:, j, :]
			post_spike_t = post_spikes[:, j, :]
			dw = stdp(pre_spike_t, post_spike_t)
			dw = dw.to('cpu').numpy()
			if dw[0, 0] != 0.:
				dw_final = dw[0, 0]
		dw_list.append(dw_final)
		stdp.reset_traces()

	dt_list = np.array(dt_list[1:])
	dw_list = np.array(dw_list[1:])
	plt.plot(dt_list, dw_list)
	plt.axhline(0, ls='--', c='black', linewidth=1)
	plt.axvline(0, ls='--', c='black', linewidth=1)
	plt.xlabel(r'$\Delta t = (t_{\mathrm{post}} - t_{\mathrm{pre}})$ (ms)')
	plt.ylabel(r'$\Delta w$')

	plt.title(f'I-STDP')
	plt.tight_layout(pad=0.5)
	plt.show()
	end = time.time()
	print('Running ', end - start, 's')
	return np.array(dt_list[1:]), np.array(dw_list[1:])


# noinspection PyTypeChecker
def fun2(dt):
	start = time.time()
	time_ = 50
	time_steps = int(time_ / dt)
	spike_train = np.eye(time_steps)
	# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'

	conn = np.ones((1, 1))
	ESTDP_ = {'lr': 1., 'tau_pre': 10., 'tau_post': 10., 'A1': 0.15, 'A2': 0.15 * 1.2, 'a1': 1., 'a2': 1., 'conn':conn}
	stdp = ESTDP(dt=dt, **ESTDP_).to(device)

	dt_list = []
	dw_list = []

	post_spikes = torch.zeros((1, time_steps, 1)).to(device)
	post_spikes[0, :, 0] = torch.tensor(spike_train[0])
	t_post = np.argmax(spike_train[0])
	for i in range(time_steps):
		t_pre = np.argmax(spike_train[-i])
		dt_list.append((t_post - t_pre) * dt)  # 计算 dt
		pre_spikes = torch.zeros((1, time_steps, 1)).to(device)
		pre_spikes[0, :, 0] = torch.tensor(spike_train[-i])
		dw_final = 0.
		for j in range(time_steps):
			pre_spike_t = pre_spikes[:, j, :]
			post_spike_t = post_spikes[:, j, :]
			dw = stdp(pre_spike_t, post_spike_t)
			dw = dw.to('cpu').numpy()
			if dw[0, 0] != 0.:
				dw_final = dw[0, 0]
		dw_list.append(dw_final)
		stdp.reset_traces()

	pre_spikes = torch.zeros((1, time_steps, 1)).to(device)
	pre_spikes[0, :, 0] = torch.tensor(spike_train[0])
	t_pre = np.argmax(spike_train[0])
	for i in range(time_steps):
		t_post = np.argmax(spike_train[i])
		dt_list.append((t_post - t_pre) * dt)
		post_spikes = torch.zeros((1, time_steps, 1)).to(device)
		post_spikes[0, :, 0] = torch.tensor(spike_train[i])
		dw_final = 0.
		for j in range(time_steps):
			pre_spike_t = pre_spikes[:, j, :]
			post_spike_t = post_spikes[:, j, :]
			dw = stdp(pre_spike_t, post_spike_t)
			dw = dw.to('cpu').numpy()
			if dw[0, 0] != 0.:
				dw_final = dw[0, 0]
		dw_list.append(dw_final)
		stdp.reset_traces()

	dt_list = np.array(dt_list[1:])
	dw_list = np.array(dw_list[1:])
	plt.plot(dt_list, dw_list)
	plt.axhline(0, ls='--', c='black', linewidth=1)
	plt.axvline(0, ls='--', c='black', linewidth=1)
	plt.xlabel(r'$\Delta t = (t_{\mathrm{post}} - t_{\mathrm{pre}})$ (ms)')
	plt.ylabel(r'$\Delta w$')
	plt.title('E-STDP')
	plt.tight_layout()
	plt.show()
	end = time.time()
	print('Running ', end - start, 's')
	return np.array(dt_list[1:]), np.array(dw_list[1:])


if __name__ == '__main__':
	dt_ = 0.1

	dt_list1, dw_list1 = fun1(dt_)
	dt_list2, dw_list2 = fun2(dt_)

	# np.savez_compressed(rf'./STDP_data2.npz',
	# 					ISTDP_dt=dt_list1, ISTDP_dw=dw_list1,
	# 					ESTDP_dt=dt_list2, ESTDP_dw=dw_list2)

	_, axs = plt.subplots(1, 1)
	axs.axhline(0, ls='--', c='black', linewidth=1)
	axs.axvline(0, ls='--', c='black', linewidth=1)
	axs.plot(dt_list1, dw_list1, label='I-STDP')
	axs.plot(dt_list2, dw_list2, label='E-STDP')
	axs.set(xlabel=r'$\Delta t = (t_{\mathrm{post}} - t_{\mathrm{pre}})$ (ms)', ylabel=r'$\Delta w$')
	axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
	plt.legend()
	plt.tight_layout(pad=0.5)
	plt.show()


