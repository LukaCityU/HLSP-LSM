# HLSP-LSM: Hybrid Long Short-Term Plasticity Liquid State Machine

This repository contains the official implementation of the paper:

"HLSP-LSM: Enhancing image recognition performance of liquid state machines via brain-inspired hybrid long short-term plasticity"

HLSP-LSM introduces a hybrid long short-term plasticity (HLSP) mechanism into the classical Liquid State Machine (LSM) architecture. Inspired by biological synaptic plasticity, HLSP-LSM integrates:

- Long-term plasticity (E-STDP & I-STDP)
- Short-term plasticity (STD)

to significantly enhance the spatiotemporal feature extraction and recognition performance of LSMs on benchmark datasets such as MNIST, FashionMNIST, and NMNIST.

-------------------------------------------------------------------------------

## Key Features

- Hybrid plasticity mechanism combining STDP (long-term) and STD (short-term)
- Modular framework with clear separation of neuron, synapse, and plasticity models
- Complete runnable training/testing example (HLSPLSM.py)
- Independent simulation examples for neuron, synapse, and plasticity modules
- Supports static and event-based datasets

-------------------------------------------------------------------------------

## Project Structure
HLSP-LSM/  
│── HLSPLSM.py                # Full training & testing example  
│── LSM_BaseClass.py          # Base class of Liquid State Machine  
│── Parameters.py             # Global configuration & hyperparameters  
│── Function.py               # Utility functions  
│── BaseModel/                # Core neuron, synapse and plasticity implementations  
│   │── NeuronModel/  
│   │   └── LIF.py            # Leaky Integrate-and-Fire neuron model  
│   │── SynapseModel/  
│   │   └── Expon.py          # Exponential decay synapse  
│   └── PlasticityModel/  
│       │── ESTDP.py          # Excitatory STDP  
│       │── ISTDP.*           # Inhibitory STDP (compiled cpython .so/.pyd)  
│       └── STD.*             # Short-term depression (compiled backend)  
│── Simulation/  
│   │── Neuron/  
│   │   └── LIF_mini_current.py  
│   │── Synapse/  
│   │   └── Expon_CUBA.py  
│   └── Plasticity/  
│       │── STDP_study_window.py  
│       │── Synapse_with_STD.py  
│       └── Synapse_with_STDP.py  
│── Datasets/  
│   │── MNIST/  
│   │── NMNIST/  
│   └── FashionMNIST/  
│── Results/                  # Output (ignored by git)  
└── README.md  

-------------------------------------------------------------------------------

## Installation

1. Create environment:  
conda create -n hlsp_lsm python=3.12  
conda activate hlsp_lsm

2. Install dependencies:  
pip install -r requirements.txt

-------------------------------------------------------------------------------

## Dataset Preparation

Datasets are not included in this repository.

Please download and place them under:

Datasets/  
│── MNIST/  
│── NMNIST/  
└── FashionMNIST/  

Each dataset directory may contain a README.md describing the expected file format.

-------------------------------------------------------------------------------

## Running HLSP-LSM (Training & Testing)

Run the full HLSP-LSM pipeline:

python HLSPLSM.py

This will:
1. Construct the Liquid State Machine
2. Load dataset
3. Apply hybrid plasticity (STDP + STD)
4. Train and evaluate the model
5. Save outputs under Results/

-------------------------------------------------------------------------------

## Module-wise Simulations

1. Neuron Model:  
python Simulation/Neuron/LIF_mini_current.py

2. Synapse Model:  
python Simulation/Synapse/Expon_CUBA.py

3. Plasticity Models:  
python Simulation/Plasticity/STDP_study_window.py  
python Simulation/Plasticity/Synapse_with_STD.py  
python Simulation/Plasticity/Synapse_with_STDP.py  

-------------------------------------------------------------------------------

## Plasticity Modules (.so / .pyd)

To improve simulation speed, the following modules are provided with precompiled backends:

ISTDP.cpython-313-darwin.so  
ISTDP.cp312-win_amd64.pyd  
STD.cpython-313-darwin.so  
STD.cp312-win_amd64.pyd  
ISTDP.pyi / STD.pyi type stubs

Python automatically chooses:
- .pyd on Windows
- .so on macOS/Linux

If the precompiled binaries do not match your Python version, please contact the authors.
The source files will be released after the paper is accepted.

-------------------------------------------------------------------------------

## Results

All experiment outputs will be saved into the Results/ directory.  
Git ignores this directory.

-------------------------------------------------------------------------------

## License

HLSP-LSM is distributed under the MIT License. See LICENSE for details.

-------------------------------------------------------------------------------

## Citation

If you use HLSP-LSM in your research, please cite (the article has been submitted):

@article{HLSPLSM2025,  
  title={HLSP-LSM: Enhancing image recognition performance of liquid state machines 
via brain-inspired hybrid long short-term plasticity},  
  author={...},  
  journal={...},  
  year={2025}  
}

-------------------------------------------------------------------------------

## Contact

For questions or discussions, please open an issue or contact:  
chaoluo2023@gmail.com  
D22092100409@cityu.edu.mo
