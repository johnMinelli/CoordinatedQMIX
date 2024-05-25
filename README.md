# MARL Coordination with QMIX

[![ArXiv](https://img.shields.io/badge/arXiv-2308.10721-b31b1b.svg)](https://arxiv.org/abs/2308.10721)

Official repository for *"CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision Making"*

## Project structure
- `core` folders contains both ppo implementation used to act in Carla environment using [carla-gym](https://github.com/johnMinelli/carla-gym/) wrapper and official implementation of CoMix used for testing in simple multi agents environments.
- `baseline` folder contains algorithms from the literature as testing bed for our method and attempts of implementation of our approach in others CTDE frameworks.
- `envs` contains environments used for test and evaluation, partially reimplemented starting from [ma-gym repository](https://github.com/koulanurag/ma-gym)

## Setup

- Setup an environment (optional) and install the requirements:

  ```
  conda create -n env python=3.9.2
  conda activate env
  pip install -r requirements.txt
  ```

## Usage
Default parameters are loaded for each environment in automatic through the params_{env_name}.yaml config file if exists. Otherwise a custom configuration can be loaded specifying the file path in `--yaml_params` parameter or specifying individual parameters in the command line.

To train:

```
python train.py --env CoMix_switch
```
- Environment names available are: `CoMix_switch`, `CoMix_predator_prey_4`, `CoMix_predator_prey_8`, `CoMix_predator_prey_16`, `CoMix_transport_1`, `CoMix_transport_2`, `CoMix_transport_4` 

- Fine tuning procedure explained in the paper, can be executed using `--fine_tune 1` and modifying the Q optimizer parameters)

To evaluate:
```
python eval.py --models_path save --model_epoch -1  -ve 1000
```
-  -1 for last checkpoint in folder
- 1000 validation episodes to get a strong average score 

# 📚 Citation

Consider giving it a ⭐ and cite our paper:

```
@article{Minelli2023Comix,
  author = {Giovanni Minelli and Mirco Musolesi},
  journal = {Transactions on Machine Learning Research},
  title={CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision-Making},
  year = {2024},
  url={https://arxiv.org/abs/2308.10721}
}
```