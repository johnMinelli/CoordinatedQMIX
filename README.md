# MARL Coordination with QMIX

Official repository for *"CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision Making"* [[ArXiv](https://arxiv.org/abs/2308.10721)]

## Project structure
The `master` branch repository provide the code used for the paper.
The `dev` branch contains all material for testing and ablation study.

## Usage
Default parameters are loaded for each environment in automatic through the params_{env_name}.yaml config file if exists. Otherwise a custom configuration can be loaded specifying the file path in `--yaml_params` parameter or specifying individual parameters in the command line.


To train

```
python train.py --env CoMix_switch
```
(fine tuning procedure explained in the paper, can be executed using `--fine_tune 1` and modifying the Q optimizer parameters)

To evaluate:
```
python eval.py --models_path save --model_epoch -1
```

### Cite
```
@misc{minelli2023comix,
      title={CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision Making}, 
      author={Giovanni Minelli and Mirco Musolesi},
      year={2023},
      eprint={2308.10721},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
