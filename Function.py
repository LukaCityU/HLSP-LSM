import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Parameters import Generate_batch_size


# --------------------------  VISUALIZATION  --------------------------
def show_liquid_state(ts, dt, duration, inp_sps, e_sps, i_sps,
					  inpE_pos_cur, inpE_neg_cur, EE_cur, IE_cur,
					  inpI_pos_cur, inpI_neg_cur, EI_cur, II_cur,  save_dir=None):
	fig = plt.figure(figsize=(10, 6))
	gs = GridSpec(6, 2)
	NE = e_sps.shape[1]
	NI = i_sps.shape[1]
	N = NE + NI

	ax1 = fig.add_subplot(gs[0:3, 0])
	t, neu_index = np.where(inp_sps)
	t = t * dt
	ax1.scatter(t, neu_index, s=0.5, c='k')
	ax1.set(xlabel=r'$t$ (ms)', ylabel='Neuron Index', title=f'Input Spikes: {np.sum(inp_sps)}',
			xlim=[-1, duration + 1])
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	ax2 = fig.add_subplot(gs[3:6, 0])
	n_e = e_sps.shape[1]
	t, neu_index = np.where(e_sps)
	t = t * dt
	ax2.scatter(t, neu_index, s=0.5, c='red')
	t, neu_index = np.where(i_sps)
	t = t * dt
	ax2.scatter(t, neu_index + n_e, s=0.5, c='blue')
	ax2.set(xlabel=r'$t$ (ms)', ylabel='Neuron Index', title=f'Liquid Spikes: {np.sum(e_sps)}, {np.sum(i_sps)}',
			xlim=[-1, duration + 1])
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)

	ax4 = fig.add_subplot(gs[0:2, 1])
	# [time, neuron_index] -> [time, ]
	inp_pos = (np.sum(inpE_pos_cur, axis=1) + np.sum(inpI_pos_cur, axis=1)) / N
	inp_neg = (np.sum(inpE_neg_cur, axis=1) + np.sum(inpI_neg_cur, axis=1)) / N
	ax4.axhline(0, linestyle='--', color=u'#ff7f0e')
	ax4.plot(ts, inp_pos, label='E', color=u'r')
	ax4.plot(ts, inp_neg, label='I', color=u'b')
	ax4.set(xlabel=r'$t$ (ms)', ylabel='Current', title='Input', xlim=[-1, duration + 1])
	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)

	ax5 = fig.add_subplot(gs[2:4, 1])
	rec_pos = (np.sum(EE_cur, axis=1) + np.sum(EI_cur, axis=1)) / N
	rec_neg = (np.sum(IE_cur, axis=1) + np.sum(II_cur, axis=1)) / N
	ax5.axhline(0, linestyle='--', color=u'#ff7f0e')
	ax5.plot(ts, rec_pos, label='E', color=u'r')
	ax5.plot(ts, rec_neg, label='I', color=u'b')
	ax5.set(xlabel=r'$t$ (ms)', ylabel='Current', title='Recurrent', xlim=[-1, duration + 1])
	ax5.spines['top'].set_visible(False)
	ax5.spines['right'].set_visible(False)

	ax6 = fig.add_subplot(gs[4:6, 1])
	tot_pos = inp_pos + rec_pos
	tot_neg = inp_neg + rec_neg
	tot = tot_pos + tot_neg
	ax6.axhline(0, linestyle='--', color=u'#ff7f0e')
	ax6.plot(ts, tot_pos, label='E', color=u'r')
	ax6.plot(ts, tot_neg, label='I', color=u'b')
	ax6.plot(ts, tot, label='E + I', color=u'g')
	ax6.set(xlabel=r'$t$ (ms)', ylabel='Current', title='Total', xlim=[-1, duration + 1])
	ax6.spines['top'].set_visible(False)
	ax6.spines['right'].set_visible(False)

	plt.tight_layout()
	if save_dir is None:
		plt.show()
	else:
		plt.savefig(save_dir)
		plt.close(fig)

# --------------------------  UTILITY FUNCTIONS  --------------------------


# --------------------------  FULL TRAINING PIPELINE  --------------------------
# Generate reservoir (liquid) states
def generate_liquid_state(net, LSM_train_set, LSM_val_set, LSM_test_set, device, time_step):
	N_rec = net.rec_size
	N_E = net.E_size
	LSM_train_loader = DataLoader(dataset=LSM_train_set, batch_size=Generate_batch_size, shuffle=False, drop_last=False)
	LSM_val_loader = DataLoader(dataset=LSM_val_set, batch_size=Generate_batch_size, shuffle=False, drop_last=False)
	LSM_test_loader = DataLoader(dataset=LSM_test_set, batch_size=Generate_batch_size, shuffle=False, drop_last=False)
	train_liquid_state = np.zeros((len(LSM_train_set), N_rec))
	train_liquid_label = np.zeros((len(LSM_train_set),))
	val_liquid_state = np.zeros((len(LSM_val_set), N_rec))
	val_liquid_label = np.zeros((len(LSM_val_set),))
	test_liquid_state = np.zeros((len(LSM_test_set), N_rec))
	test_liquid_label = np.zeros((len(LSM_test_set),))

	net.eval()
	net.reset_state()
	with torch.no_grad():
		print('Generating reservoir states for train sets...')
		for i, (img, label) in enumerate(LSM_train_loader):
			img = img.float().to(device)
			_, e_sps, i_sps = net(img, record=False, STDP=False, time_step=time_step)

			train_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, :N_E] = e_sps.cpu().numpy()
			train_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, N_E:] = i_sps.cpu().numpy()
			train_liquid_label[i * Generate_batch_size: (i + 1) * Generate_batch_size] = label.cpu().numpy()
			print(f"Progress bar [{(i + 1) * Generate_batch_size}/{len(LSM_train_set)}]")

		print('Generating reservoir states for val sets...')
		for i, (img, label) in enumerate(LSM_val_loader):
			img = img.float().to(device)
			_, e_sps, i_sps = net(img, record=False, STDP=False, time_step=time_step)

			val_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, :N_E] = e_sps.cpu().numpy()
			val_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, N_E:] = i_sps.cpu().numpy()
			val_liquid_label[i * Generate_batch_size: (i + 1) * Generate_batch_size] = label.cpu().numpy()
			print(f"Progress bar [{(i + 1) * Generate_batch_size}/{len(LSM_val_set)}]")

		print('Generating reservoir states for test sets...')
		for i, (img, label) in enumerate(LSM_test_loader):
			img = img.float().to(device)
			_, e_sps, i_sps = net(img, record=False, STDP=False, time_step=time_step)

			test_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, :N_E] = e_sps.cpu().numpy()
			test_liquid_state[i * Generate_batch_size: (i + 1) * Generate_batch_size, N_E:] = i_sps.cpu().numpy()
			test_liquid_label[i * Generate_batch_size: (i + 1) * Generate_batch_size] = label.cpu().numpy()
			print(f"Progress bar [{(i + 1) * Generate_batch_size}/{len(LSM_test_set)}]")

	return train_liquid_state, train_liquid_label, val_liquid_state, val_liquid_label, test_liquid_state, test_liquid_label

# Collect and concatenate reservoir states for readout
class CustomDataset(Dataset):
	def __init__(self, data, labels):
		self.data = torch.tensor(data)
		self.labels = torch.tensor(labels, dtype=torch.long)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

# Readout layer
class LinearModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LinearModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.dropout(x)
		return self.linear(x)

# Train readout layer
def train_readout(train_liquid_state, train_liquid_label, val_liquid_state, val_liquid_label,
                  test_liquid_state, test_liquid_label, input_dim, output_dim, device, save_dir=None):

	readout = LinearModel(input_dim=input_dim, output_dim=output_dim).to(device)

	r_train_set = CustomDataset(train_liquid_state, train_liquid_label)
	r_val_set = CustomDataset(val_liquid_state, val_liquid_label)
	r_test_set = CustomDataset(test_liquid_state, test_liquid_label)
	r_train_loader = DataLoader(r_train_set, batch_size=512, shuffle=True, drop_last=False)
	r_val_loader = DataLoader(r_val_set, batch_size=10000, shuffle=False, drop_last=False)
	r_test_loader = DataLoader(r_test_set, batch_size=10000, shuffle=False, drop_last=False)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(readout.parameters(), lr=1e-3, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

	num_epochs = 50
	train_loss = []
	val_acc = []
	for epoch in range(num_epochs):
		readout.train()
		for i, (batch_data, batch_labels) in enumerate(r_train_loader):
			optimizer.zero_grad()
			batch_data = batch_data.float().to(device)
			batch_labels = batch_labels.to(device)
			outputs = readout(batch_data)
			loss = criterion(outputs, batch_labels)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		readout.eval()
		running_loss = 0.0
		with torch.no_grad():
			for i, (batch_data, batch_labels) in enumerate(r_val_loader):
				batch_data = batch_data.float().to(device)
				batch_labels = batch_labels.to(device)
				outputs = readout(batch_data)
				loss = criterion(outputs, batch_labels)
				predicted_labels = torch.argmax(outputs, dim=1)
				accuracy = (predicted_labels == batch_labels).float().mean().cpu().numpy()
				val_acc.append(float(accuracy * 100.))
				print('Validate —— Epoch ' + str(epoch + 1) + ' Iteration ' + str(i + 1) + ' ACC: ' + str(accuracy) + ' Loss: ' + str(loss.item()))
				running_loss += loss.item()
			running_loss /= len(r_train_loader)
			scheduler.step(running_loss)
			current_lr = scheduler.get_last_lr()[0]
			print(f'Current Learning Rate: {current_lr:.6f}')

	readout.eval()
	accuracy = 0.
	y_true = []
	y_pred = []
	with torch.no_grad():
		for i, (batch_data, batch_labels) in enumerate(r_test_loader):
			batch_data = batch_data.float().to(device)
			batch_labels = batch_labels.to(device)
			outputs = readout(batch_data)
			predicted_labels = torch.argmax(outputs, dim=1)
			accuracy = (predicted_labels == batch_labels).float().mean().cpu().numpy()
			y_true.extend(batch_labels.cpu().numpy())
			y_pred.extend(predicted_labels.cpu().numpy())
		print(f'Test Accuracy: {accuracy * 100.:.2f}')

	fig, axs = plt.subplots(1, 2, figsize=(6, 2))
	axs[0].plot(np.array(train_loss), label='Train Loss')
	axs[0].set(ylabel='Loss', xlabel='Iteration')
	axs[0].spines['top'].set_visible(False)
	axs[0].spines['right'].set_visible(False)
	axs[0].legend()
	axs[1].plot(np.array(val_acc), label='Validate Accuracy')
	axs[1].set(ylabel='Accuracy (%)', xlabel='Epoch')
	axs[1].spines['top'].set_visible(False)
	axs[1].spines['right'].set_visible(False)
	axs[1].legend()
	fig.suptitle(f'Test Accuracy: {accuracy * 100.:.2f}')
	fig.tight_layout()
	if save_dir is None:
		plt.show()
	else:
		plt.savefig(save_dir)
		plt.close(fig)

	return accuracy * 100.
