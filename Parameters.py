import os


device = 'cpu'
random_seed = [667999, 969449]
w_in = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


DATASET_BASE = {'DATASET': 'SIMULATION', 'input_size': 100, 'num_classes': 2,
                'num_train_init': 100, 'num_test': 10, 'num_val': 10, 'num_analysis': 2}
# DATASET_BASE = {'DATASET': 'SIMULATIONSPIKE', 'input_size': 100, 'num_classes': 2,
#                 'num_train_init': 100, 'num_test': 10, 'num_val': 10, 'num_analysis': 2}

# DATASET_BASE = {'DATASET': 'MNIST', 'input_size': 784, 'num_classes': 10,
#                 'num_train_init': 60000, 'num_test': 10000, 'num_val': 10000, 'num_analysis': 200}
# DATASET_BASE = {'DATASET': 'FashionMNIST', 'input_size': 784, 'num_classes': 10,
#                 'num_train_init': 60000, 'num_test': 10000, 'num_val': 10000, 'num_analysis': 200}
# DATASET_BASE = {'DATASET': 'NMNIST', 'input_size': 2312, 'num_classes': 10,
#                 'num_train_init': 60000, 'num_test': 10000, 'num_val': 10000, 'num_analysis': 200}


MODEL = 'EISTDPSTP'
# MODEL = 'EISTDP'
# MODEL = 'ESTDP'
# MODEL = 'ESTDPSTP'


DATASET = DATASET_BASE['DATASET']
path = os.path.join('.', 'Results', DATASET)
root_dir = os.path.join('.', 'Results', f'{DATASET}_{MODEL}')


input_size = DATASET_BASE['input_size']
rec_size = 100
num_train_init = DATASET_BASE['num_train_init']
num_test = DATASET_BASE['num_test']
num_val = DATASET_BASE['num_val']
num_train = num_train_init - num_val
num_analysis = DATASET_BASE['num_analysis']
num_classes = DATASET_BASE['num_classes']


N_CHECKPOINT = 2
STDP_batch_size = 10
Generate_batch_size = 10000