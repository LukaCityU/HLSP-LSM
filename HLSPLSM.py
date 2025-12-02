from torch.utils.data import random_split

import time
import math
import h5py

from LSM_BaseClass import LSMEISTDP, LSMParameters, LSMEISTDPSTP, LSMESTDP, LSMESTDPSTP
from Function import *
from Parameters import *


class RandomDataset(Dataset):
	def __init__(self, num_sample, input_dim, num_classes_):
		self.num_sample = num_sample
		self.input_dim = input_dim
		self.num_classes = num_classes_

		# Generate random input data (floats between 0 and 1)
		self.data = torch.rand(num_sample, input_dim)
		# Generate random labels (integers between 0 and 9)
		self.labels = torch.randint(0, num_classes_, (num_sample,))

		self.ndim = self.data.ndim
		self.time_dim = None

	def __len__(self):
		return self.num_sample

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


class RandomSpikeDataset(Dataset):
	def __init__(self, num_sample, input_dim, num_classes_, duration=250, dt=1.0, frequency=100):
		self.num_sample = num_sample
		self.input_dim = input_dim
		self.num_classes = num_classes_

		time_steps = int(duration / dt)

		# Generate random input data (floats between 0 and 1)
		random_values = torch.rand(num_sample, time_steps, input_dim)
		self.data = (random_values < frequency / 1000 * dt).float()
		# Generate random labels (integers between 0 and 9)
		self.labels = torch.randint(0, num_classes_, (num_sample,))

		self.ndim = self.data.ndim
		self.time_dim = time_steps

	def __len__(self):
		return self.num_sample

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


class MNIST(Dataset):
	def __init__(self, data_path):
		loaded_arrays = np.load(data_path)
		x = loaded_arrays['x']     # [num, inp_dim]
		sum_x = np.sum(x, axis=1)  # [num]
		normed_x = np.divide(x, sum_x[:, np.newaxis])  # [num, inp_dim]
		rescale_factor = np.amax(normed_x)
		self.images = normed_x / rescale_factor
		self.labels = loaded_arrays['y']

		self.num_sample = self.images.shape[0]
		self.ndim = x.ndim
		self.time_dim = None

	def __getitem__(self, index):
		return torch.tensor(self.images[index]), torch.tensor(self.labels[index])

	def __len__(self):
		return self.num_sample


class NMNIST(Dataset):
	def __init__(self, data_path=None):
		d = h5py.File(data_path, 'r')
		image, label = d[list(d.keys())[0]], d[list(d.keys())[1]]  # [num, time_step, inp_dim]

		self.images = torch.from_numpy(np.array(image))
		self.labels = torch.from_numpy(np.array(label))

		self.num_sample = self.images.size(0)
		self.ndim = self.images.ndim
		self.time_dim = self.images.size(1)

	def __getitem__(self, index):
		img, target = self.images[index], self.labels[index]
		return img, target

	def __len__(self):
		return self.num_sample


def HLSPLSM(dt, main_net, init_train_set, LSM_test_set, seed=10086, win=1.5, root=root_dir):
	checkpoint_dir = os.path.join(root, 'checkpoint')
	state_dir = os.path.join(root, 'state')
	analysis_dir = os.path.join(root, 'analysis')
	result_dir = os.path.join(root, 'result')

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(state_dir):
		os.makedirs(state_dir)
	if not os.path.exists(analysis_dir):
		os.makedirs(analysis_dir)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# region (1) Parameter setup
	N_rec = rec_size
	P = LSMParameters(input_size=input_size, rec_size=rec_size, dt=dt, seed=seed, w_in=win)
	# endregion

	# region (2) Network construction
	net = main_net(parameter=P.parameter)
	net.to(device)
	# endregion

	# region (3) Load dataset
	print('Load dataset...')
	# Get dataset dimensionality and time dimension size
	# Dimensionality = 2 → static data (needs rate/frequency encoding); = 3 → already spiking data
	data_ndim = init_train_set.ndim
	data_time_dim = init_train_set.time_dim  # Time dimension size: None for non-spiking datasets
	# Use stratified sampling to split training set
	generator = torch.Generator().manual_seed(seed)
	LSM_train_set, LSM_val_set = random_split(init_train_set, [num_train, num_val], generator=generator)
	# endregion

	# region (4) STDP training phase
	simulation_time = 50  # ms
	indices = np.arange(simulation_time / dt)  # time step
	ts = indices * dt
	STDP_train_loader = DataLoader(dataset=LSM_train_set, batch_size=STDP_batch_size, shuffle=True, drop_last=False)

	print('STDP training phase...')
	stop = len(LSM_train_set)  # Train using all samples
	stop = math.ceil(stop / (STDP_batch_size * N_CHECKPOINT)) * (STDP_batch_size * N_CHECKPOINT)
	net.eval()
	with torch.no_grad():
		for i, (img, label) in enumerate(STDP_train_loader):
			# If dataset is already spiking, resample/time-stretch to simulation_time length
			if data_ndim == 3:
				indd = np.sort(np.random.choice(data_time_dim, size=int(simulation_time / dt), replace=False))
				img = img[:, indd, :]

			img = img.float().to(device)
			if (i + 1) % ((stop / STDP_batch_size) / N_CHECKPOINT) == 0 or i == 0:
				print(f"Progress bar [{(i + 1) * STDP_batch_size}/{stop}]")
				inp_sps, e_sps, i_sps, NGR, SGR = net(img, record=True, STDP=True, time_step=int(simulation_time / dt))

				show_liquid_state(ts=ts, dt=dt, duration=simulation_time,
				                  inp_sps=inp_sps.cpu().numpy()[0], e_sps=NGR.S_E[0], i_sps=NGR.S_I[0],
				                  inpE_pos_cur=SGR.inpE_pos[0], inpE_neg_cur=SGR.inpE_neg[0],
				                  EE_cur=SGR.EE[0], IE_cur=SGR.IE[0],
				                  inpI_pos_cur=SGR.inpI_pos[0], inpI_neg_cur=SGR.inpI_neg[0],
				                  EI_cur=SGR.EI[0], II_cur=SGR.II[0],
				                  save_dir=os.path.join(state_dir, f'state_{(i + 1) * STDP_batch_size}.png'))

			else:
				net(img, record=False, STDP=True, time_step=int(simulation_time / dt))

			if ((i + 1) * STDP_batch_size) >= stop:
				break

	state = {'net': net.state_dict()}
	torch.save(state, os.path.join(checkpoint_dir, 'final_checkpoint.pth'))
	# endregion

	# region (5) Collect liquid state vectors and train readout
	simulation_time = 250  # ms
	time_step = int(simulation_time / dt)
	train_liquid_state, train_liquid_label, val_liquid_state, val_liquid_label, test_liquid_state, test_liquid_label = generate_liquid_state(
		net, LSM_train_set, LSM_val_set, LSM_test_set, device, time_step)
	train_readout(train_liquid_state, train_liquid_label, val_liquid_state, val_liquid_label,
	              test_liquid_state, test_liquid_label, N_rec, num_classes, device,
	              save_dir=os.path.join(result_dir, 'TrainTest_final.png'))
	# endregion

	print(f'done')


if __name__ == '__main__':
	start_ = time.time()
	dt_ = 1.

	if MODEL == 'EISTDP':
		MAIN_NET = LSMEISTDP
	elif MODEL == 'EISTDPSTP':
		MAIN_NET = LSMEISTDPSTP
	elif MODEL == 'ESTDP':
		MAIN_NET = LSMESTDP
	elif MODEL == 'ESTDPSTP':
		MAIN_NET = LSMESTDPSTP
	else:
		raise ValueError("Model type error.")

	if DATASET == 'SIMULATION':
		init_train_set_ = RandomDataset(num_sample=num_train_init, input_dim=input_size, num_classes_=num_classes)
		LSM_test_set_ = RandomDataset(num_sample=num_test, input_dim=input_size, num_classes_=num_classes)
	elif DATASET == 'SIMULATIONSPIKE':
		init_train_set_ = RandomSpikeDataset(num_sample=num_train_init, input_dim=input_size, num_classes_=num_classes)
		LSM_test_set_ = RandomSpikeDataset(num_sample=num_test, input_dim=input_size, num_classes_=num_classes)
	elif DATASET == 'MNIST':
		init_train_set_ = MNIST(os.path.join(path, 'train.npz'))
		LSM_test_set_ = MNIST(os.path.join(path, 'test.npz'))
	elif DATASET == 'FashionMNIST':
		init_train_set_ = MNIST(os.path.join(path, 'train.npz'))
		LSM_test_set_ = MNIST(os.path.join(path, 'test.npz'))
	elif DATASET == 'NMNIST':
		init_train_set_ = NMNIST(os.path.join(path, 'train.mat'))
		LSM_test_set_ = NMNIST(os.path.join(path, 'test.mat'))
	else:
		raise ValueError("Dataset type error.")

	HLSPLSM(dt_, MAIN_NET, init_train_set_, LSM_test_set_)