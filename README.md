# MPC-CRL: Chance-Aware Lane Change with High-Level Model Predictive Control Through Curriculum Reinforcement Learning
Author: Yubin Wang

Code for [*Chance-Aware Lane Change with High-Level Model Predictive Control Through Curriculum Reinforcement Learning*](https://arxiv.org/abs/2303.03723) (IROS 2023 Submission)

## Requirements
* Python 3.8.1 
* [PyTorch](http://pytorch.org/)
* [Wandb](https://wandb.ai)
* [CasADi](https://web.casadi.org/), version: 3.5.5

The versions are just what I used and not necessarily strict requirements.

## How to Run

### Train

Train our method MPC-CRL:
```shell
python train_CRL_MPC.py
```

Train baseline MPC-vanilla RL:
```shell
python train_standardRL_MPC.py
```

### Evaluate

Evaluate the performance of trained model:
```shell
python eval_learningMPC.py
```

## More

Under construction...

## Note

Please consider citing the our paper if useful :

```bibtex
@article{wang2023chance,
  title={Chance-Aware Lane Change with High-Level Model Predictive Control Through Curriculum Reinforcement Learning},
  author={Wang, Yubin and Li, Yulin and Ghazzai, Hakim and Massoud, Yehia and Ma, Jun},
  journal={arXiv preprint arXiv:2303.03723},
  year={2023}
}
```