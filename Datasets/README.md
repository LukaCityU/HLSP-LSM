# Datasets

This directory contains all datasets used in the HLSP-LSM experiments.  
Currently, three datasets are supported:

- MNIST
- FashionMNIST
- NMNIST

Raw datasets are **not** included in this repository.  
You need to download the original datasets and convert them into the formats described below.

-------------------------------------------------------------------------------

## 1. Directory Structure

The expected directory structure is:

Datasets/  
│── MNIST/  
│   │── train.npz  
│   └── test.npz  
│── FashionMNIST/  
│   │── train.npz  
│   └── test.npz  
└── NMNIST/  
    │── train.mat  
    └── test.mat  

Each subdirectory may also contain additional helper scripts (e.g., preprocessing scripts) and README files if needed.

-------------------------------------------------------------------------------

## 2. MNIST and FashionMNIST

### 2.1 File format

For both **MNIST** and **FashionMNIST**, the dataset should be stored as:

- `train.npz`
- `test.npz`

Each `.npz` file must contain two arrays:

- `x`: shape `[num_samples, input_dim]`
- `y`: shape `[num_samples]`

Where:

- `x` is the flattened input (e.g., for 28×28 images, `input_dim = 784`)
- `y` stores the integer class labels

A typical loading class (used in this project) is:

```python
class MNIST(Dataset):
    def __init__(self, data_path):
        loaded_arrays = np.load(data_path)
        x = loaded_arrays['x']     # [num, inp_dim]

        # Per-sample normalization
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
````

You are free to choose your own preprocessing pipeline as long as you end up with the above `.npz` format.

### 2.2 Summary

* Place converted files under:

  * `Datasets/MNIST/train.npz`, `Datasets/MNIST/test.npz`
  * `Datasets/FashionMNIST/train.npz`, `Datasets/FashionMNIST/test.npz`
* Each `.npz` file contains:

  * `x`: `[num_samples, input_dim]`
  * `y`: `[num_samples]`

---

## 3. NMNIST

### 3.1 File format

For **NMNIST**, the dataset should be stored as:

* `train.mat`
* `test.mat`

Both files are expected to be HDF5-compatible `.mat` files, and they will be loaded using `h5py`.

Each `.mat` file must contain:

* `image`: shape `[num_samples, time_steps, input_dim]`
* `label`: shape `[num_samples]`

The corresponding dataset class used in this project is:

```python
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
```

### 3.2 Preprocessing hint

You can use the **SpikingJelly** framework to help download and convert the raw NMNIST dataset into the desired format.

Reference:
[https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

The typical pipeline is:

1. Download NMNIST using spikingjelly or your own script
2. Convert the event-based representation into tensors of shape `[num_samples, time_steps, input_dim]`
3. Save them into `train.mat` and `test.mat` with datasets:

   * `image`: `[num_samples, time_steps, input_dim]`
   * `label`: `[num_samples]`

### 3.3 Example preprocessing script (placeholder)

Below is a placeholder code block for your NMNIST preprocessing script.  
You can paste your actual script here for future reference:

```python
from spikingjelly.datasets.n_mnist import NMNIST
import numpy as np
import h5py

root_dir = r'./'
frames = 250
train_set = NMNIST(root_dir, train=True, data_type='frame', frames_number=frames, split_by='number')
train_image = np.zeros((60000, frames, 2312), dtype=np.uint8)
train_label = np.zeros((60000, ), dtype=np.uint8)
for n in range(len(train_set)):
    f, l = train_set[n]
    f = f.reshape(frames, 2312).astype(np.uint8)
    train_image[n, :-1, :] = f[:-1, :]
    train_label[n] = l
with h5py.File(root_dir + r'/train_' + str(frames) + '.mat', 'w') as f:
    f.create_dataset('image', data=train_image)
    f.create_dataset('label', data=train_label)
```

---

## 4. General Notes

* The dataset loaders in this project assume that the files strictly follow the formats described above.
* If you encounter any issues with data loading, please check:

  * The shape and dtype of `x`, `y`, `image`, and `label`
  * The keys stored in `.npz` / `.mat` files
* You are encouraged to keep your preprocessing scripts in this directory (or subdirectories) to ensure full reproducibility.
