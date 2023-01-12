import torch
from torch import nn
from typing import List, Type
#import torchsnooper


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.Tanh #nn.LeakyReLU
) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
        
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    
    return nn.Sequential(*modules)


class DNN(nn.Module):
    #@torchsnooper.snoop()
    def __init__(self, input_dim: int, output_dim: int,  net_arch: List[int], model_togpu=False, device='cpu'):
        
        super().__init__()
        if model_togpu:
            self.high_policy = create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch).to(device)
        else:
            self.high_policy = create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=net_arch)

    def forward(self, obs):
            mean = obs.mean()
            std = obs.std()
            normalized_obs = (obs - mean)/std
            z = self.high_policy(normalized_obs)
            #z = normalized_z*std + mean
            
            return z

    def compute_loss(self, reward_grad, z):
        return torch.matmul(reward_grad, z)

