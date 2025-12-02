import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from BaseModel.NeuronModel import LIF
from BaseModel.SynapseModel import Expon
from BaseModel.PlasticityModel import STD


def fun1(dt):
    # Input: random Poisson-like spike train fed to paired synapses (with vs. without short-term plasticity)
    sim_time = 100  # ms
    time_steps = int(sim_time / dt)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Fixed input spike trains, shape: [batch, time, neuron]
    temp = [int(i / dt) for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
    spike_time = np.array(temp)
    spike_data = np.zeros((1, time_steps, 1))
    spike_data[:, spike_time, :] = 1

    std = STD(in_dim=1, dt=dt, tau_d=50, U=0.4)

    # Synapse
    w = np.ones((1, 1)) * 1000
    conn = np.ones((1, 1))
    syn1 = Expon(w, conn, 1., 1., dt).to(device)
    syn2 = Expon(w, conn, 1., 1., dt).to(device)

    # Neuron
    v_init = np.zeros((1,))
    param = {'V_init': v_init, 'V_reset': 0., 'V_th': 20., 'tau_m': 64., 't_ref': 2.}
    neu1 = LIF(size=1, dt=dt, **param)
    neu2 = LIF(size=1, dt=dt, **param)

    # Record
    V = np.zeros((1, time_steps, 2))
    PSC1 = np.zeros((time_steps,))
    PSC2 = np.zeros((time_steps,))
    with torch.no_grad():
        for t in range(time_steps):
            s_pre = torch.tensor(spike_data[:, t, :]).float().to(device)
            input_ = std(s_pre)  # STD module takes spikes as input and outputs depression-modulated current
            I1 = syn1(input_)
            I2 = syn2(s_pre)
            neu1(I1)
            neu2(I2)
            PSC1[t] = I1.to('cpu').numpy()[0, 0]
            PSC2[t] = I2.to('cpu').numpy()[0, 0]
            if device == 'cpu':
                V[:, t, :1] = neu1.memory_voltage.detach().numpy()
                V[:, t, 1:] = neu2.memory_voltage.detach().numpy()
            else:
                V[:, t, :1] = neu1.memory_voltage.detach().cpu().numpy()
                V[:, t, 1:] = neu2.memory_voltage.detach().cpu().numpy()

    x = np.arange(time_steps) * dt
    fig, axs = plt.subplots(3, 2, sharex='all')
    elements = np.where(spike_data[0, :, 0:1] > 0.)
    index = elements[1]
    time = x[elements[0]]
    axs[0, 0].plot(time, index, '.k', markersize=4)
    axs[0, 0].set(title='Expon + STP', yticks=[])
    axs[0, 1].plot(time, index, '.k', markersize=4)
    axs[0, 1].set(title='Expon', yticks=[])

    axs[1, 0].plot(x, PSC1)
    axs[1, 0].set(ylabel='PSC')
    axs[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%4d'))
    axs[1, 1].plot(x, PSC2)
    axs[1, 1].set(ylabel='PSC')
    axs[1, 1].sharey(axs[1, 0])

    voltage1 = V[0, :, 0]
    axs[2, 0].plot(x, voltage1)
    axs[2, 0].set(ylabel='post.V', xlabel='time')
    axs[2, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%4d'))
    voltage2 = V[0, :, 1]
    axs[2, 1].plot(x, voltage2)
    axs[2, 1].set(ylabel='post.V', xlabel='time')
    axs[2, 1].sharey(axs[2, 0])

    plt.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.5)
    plt.show()
    print('done')


if __name__ == '__main__':
    dt_ = 1.
    fun1(dt_)
