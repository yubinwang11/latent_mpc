# Latent-MPC
Author: Yubin Wang

## Requirements
* Python 3.7.1 
* [PyTorch](http://pytorch.org/)
* [Wandb](https://wandb.ai)
* [CasADi](https://web.casadi.org/) 3.5.5
* Unreal Engine 4.26
* Carla 0.9.12

## Train the Models

```shell
python main.py 
```

## Evaluate the performance 
Replace the model index with the model you trained to evaluate the performance:
```shell
python main.py --eval True --Loadmodel True --ModelIdex 35000
```

## Citations

Please consider citing the our paper if useful :

```bibtex
@article{wang2023learning,
  title={Learning the References of Online Model Predictive Control for Urban Self-Driving},
  author={Wang, Yubin and Peng, Zengqi and Xie, Yusen and Li, Yulin and Ghazzai, Hakim and Ma, Jun},
  journal={arXiv preprint arXiv:2308.15808},
  year={2024}
}
```

```bibtex
@article{wang2023chance,
  title={Chance-Aware Lane Change with High-Level Model Predictive Control Through Curriculum Reinforcement Learning},
  author={Wang, Yubin and Li, Yulin and Peng, Zengqi and Ghazzai, Hakim and Ma, Jun},
  journal={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## References
|[High-MPC](https://github.com/uzh-rpg/high_mpc)| |[Gym-CARLA](https://github.com/cjy1992/gym-carla)| |[SAC-Continuous-Pytorch ](https://github.com/XinJingHao/SAC-Continuous-Pytorch)|

