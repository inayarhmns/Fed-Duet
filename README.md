# Fed-Duet: Dual Expert-Orchestrated Framework for Continual Federated Vision-Language Learning



This is the official PyTorch implementation for the paper **"Fed-Duet: Dual Expert-Orchestrated Framework for Continual Federated Vision-Language Learning"**.

---

## Abstract

Pretrained vision-language models (VLMs), such as CLIP, have shown promise in federated learning (FL) by bringing strong multimodal representations to edge devices. 
However, continual adaptation remains a core challenge in practical federated settings, where task distributions evolve over time and data remain non-IID across clients. In this emerging area, recent works adopt parameter-efficient fine-tuning (PEFT) as a lightweight way to reduce communication overhead, yet they fail to preserve satisfactory performance under continual learning conditions. Meanwhile, traditional federated continual learning (FCL) methods lack the capacity to maintain cross-modal alignment crucial to VLM performance.
We introduce **Fed-Duet**, a novel **Du**al **E**xper**t**-orchestrated framework for efficient federated continual learning in vision-language models. Fed-Duet features a dual-expert adaptation mechanism, combining server-coordinated semantic prompts with client-personalized modular adapters. 
These pathways are dynamically fused via a cross-attention mechanism, enabling effective knowledge transfer while preserving multimodal alignment and mitigating forgetting. We evaluate Fed-Duet across multiple challenging continual learning tasks in federated vision-language settings and demonstrate that it achieves superior performance and stability compared to existing approaches. Our work highlights the importance of coordinated expert composition in enabling scalable and robust multimodal continual learning.

## Framework

The core of FedDuet is its dual-channel architecture, which processes information through two complementary pathways before dynamically integrating their outputs.

| <img src="FedDuet_Training_Overview.png" width="100%"> | <img src="Client_Local_Training.png" width="100%"> |
| :---: | :---: |
| **(a) FedDuet Training Overview** | **(b) Client Local Training** |

**Figure 1: The architecture of Fed-Duet.**
**(a) FedDuet Overview, Interaction between Clients and Central Server.** The server-side *Federated Knowledge Orchestrator* maintains a Knowledge Repository and employs an adaptive gate to dispatch Shared Semantic Experts based on client features.
**(b) Detailed Local Training.** The client-side *Dual-Expert Duet* adapts via two complementary pathways: the *Semantic Pathway* fuses Local and Shared experts via Cross-Attention Gating for semantic guidance, while the *Parametric Pathway* fine-tunes adapters for feature specialization.

## Getting Started

### 1. Installation

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/FedDuet.git
cd FedDuet
```

We recommend creating a virtual environment to manage dependencies:
```bash
# Create a virtual environment
conda create -n FedDuet python=3.9
# Activate the environment
conda activate FedDuet
```

Install the required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**Note on PyTorch**: The `requirements.txt` file specifies PyTorch versions. If you need a different version that matches your specific CUDA setup, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command.

### 2. Dataset Preparation

This project uses several datasets for evaluation, including CIFAR-100, Tiny ImageNet, DomainNet, Flowers102, OxfordPets, Food101, Caltech101, and DTD.

Please download the datasets and place them in a directory of your choice. You will need to specify the path to your dataset directory in the corresponding configuration files located in `cil/configs/`.

<!-- TODO: Add more specific instructions on dataset structure if necessary -->
```
/path/to/your/datasets/
├── cifar-100-python/
├── tiny-imagenet-200/
├── domain_net/
├── flowers-102/
├── oxford-iiit-pet/
├── food-101/
├── caltech101/
└── dtd/
```

Update the `dataset_root` variable in the `.yaml` configuration files to point to `/path/to/your/datasets/`.




## Running Experiments

Before running, switch into the `cil` directory:
```bash
cd cil
```

The main script for running experiments is `cil/main.py`, which is managed via `cil/run.sh`. This project uses [Hydra](https://hydra.cc/) for configuration management, allowing for flexible experiment setups.

### Running a Single Experiment

To run a single experiment, you can execute the `main.py` script directly and specify a configuration file. For example, to run the CIFAR-100 experiment:

```bash
python main.py --config-path=configs --config-name=CIFAR_100_FedDuet_Incremental_10_iid
```

### Customizing Runs via Command Line

Thanks to Hydra, you can easily override any parameter from the configuration file directly through the command line. This is useful for quick tests without modifying the original `.yaml` files.

For example, to change the number of communication rounds and the learning rate:
```bash
python main.py --config-path=configs --config-name=CIFAR_100_FedDuet_Incremental_10_iid com=20 lr=0.0001
```

### Running Multiple Experiments

The `cil/run.sh` script is pre-configured to run multiple experiments sequentially. You can uncomment or modify the desired configuration paths and names within the script to reproduce specific results from our paper.

To run the entire suite of experiments defined in the script:
```bash
bash run.sh
```

## Project Structure

Here is an overview of the key directories in this project:

```
Fed-Duet/
├── cil/
│   ├── clip/                 # Contains the implementation of CLIP and related components like MoE_Adapters.
│   ├── configs/              # Hydra configuration files (.yaml) for all experiments.
│   ├── continual_clip/       # Core logic for federated continual learning, including FedDuet and other compared methods.
│   ├── class_orders/         # Defines the class order for class-incremental learning scenarios.
│   ├── main.py               # The main entry point for running experiments.
│   └── run.sh                # A helper script to run multiple experiments in sequence.
├── requirements.txt        # A list of Python packages required to run the project.
└── Readme.md               # This file.
```

## Acknowledgement

Our implementation is based on the open-source project 
[MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL). We thank the authors for sharing their codes.


## Code explanation





