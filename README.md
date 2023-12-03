# Latent-MPC
Author: Yubin Wang

## Requirements
* Python 3.7.1 
* [PyTorch](http://pytorch.org/)
* [Wandb](https://wandb.ai)
* [CasADi](https://web.casadi.org/) 3.5.5
* Unreal Engine 4.26
* Carla 0.9.12

The versions are just what I used and not necessarily strict requirements.

## Train the Models

Use wandb to monitor the training
```shell
python main.py 
```

Train the models without monitoring:
```shell
python main.py --wandb False
```

## Evaluate the performance 
Replace the model index with the model your trianed to evaluate the performance:
```shell
python main.py --wandb False --eavl True --Loadmodel True --ModelIdex 35000
```

## Reference

Please consider citing the our paper if useful :

```bibtex
@article{wang2023learning,
  title={Learning the References of Online Model Predictive Control for Urban Self-Driving},
  author={Wang, Yubin and Peng, Zengqi and Ghazzai, Hakim and Ma, Jun},
  journal={arXiv preprint arXiv:2308.15808},
  year={2023}
}
```

```bibtex
@article{wang2023chance,
  title={Chance-Aware Lane Change with High-Level Model Predictive Control Through Curriculum Reinforcement Learning},
  author={Wang, Yubin and Li, Yulin and Ghazzai, Hakim and Massoud, Yehia and Ma, Jun},
  journal={arXiv preprint arXiv:2303.03723},
  year={2023}
}
```

