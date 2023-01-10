

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models


class HyperNet(nn.Module):

    def __init__(self, hidden_size=256, ray_hidden_dim=100, out_dim=1,n_hiddens = 3, n_tasks=4):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.n_tasks = n_tasks
        self.hidden_size = hidden_size
        # Init model
        self.ray_mlp = nn.Sequential(
            nn.Linear(self.n_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )
        self.Dropout = nn.Dropout(0.05)
        # Hidden layers
        #for i in range(n_hiddens):
        self.hidden_0_weights = nn.Linear(ray_hidden_dim, self.hidden_size*14) #21x256
        self.hidden_0_bias = nn.Linear(ray_hidden_dim, self.hidden_size)
        for i in range(1,n_hiddens):
            setattr(self, f"hidden_{i}_weights", nn.Linear(ray_hidden_dim, self.hidden_size*self.hidden_size)) #256x256
            setattr(self, f"hidden_{i}_bias", nn.Linear(ray_hidden_dim, self.hidden_size))
        # FC layers
        """for i in range(4):
          setattr(self, f"task_{i}_0_weights", nn.Linear(ray_hidden_dim, 256*self.hidden_size))
          setattr(self, f"task_{i}_0_bias", nn.Linear(ray_hidden_dim, 256))
          setattr(self, f"task_{i}_1_weights", nn.Linear(ray_hidden_dim, 64*256))
          setattr(self, f"task_{i}_1_bias", nn.Linear(ray_hidden_dim, 64))
          setattr(self, f"task_{i}_2_weights", nn.Linear(ray_hidden_dim, 1*64))
          setattr(self, f"task_{i}_2_bias", nn.Linear(ray_hidden_dim, 1))
        for i in range(4, 7):
          setattr(self, f"task_{i}_weights", nn.Linear(ray_hidden_dim, 1*self.hidden_size))
          setattr(self, f"task_{i}_bias", nn.Linear(ray_hidden_dim, 1))"""
        setattr(self, "fc_weights", nn.Linear(ray_hidden_dim, self.n_tasks*self.hidden_size))
        setattr(self, "fc_bias", nn.Linear(ray_hidden_dim, self.n_tasks))
    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if 'task' not in n])

    def forward(self, ray):
        features = self.ray_mlp(ray) #100
        out_dict = {}
        layer_types = ["hidden"]
        for i in layer_types:
            n_layers = self.n_hiddens

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(features)
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(features)

        """for i in range(4):
          for j in range(3):
            out_dict[f"task_{i}_{j}.weights"] = getattr(self, f"task_{i}_{j}_weights")(features)
            out_dict[f"task_{i}_{j}.bias"] = getattr(self, f"task_{i}_{j}_bias")(features)
        for i in range(4, 7):
            out_dict[f"task_{i}.weights"] = getattr(self, f"task_{i}_weights")(features)
            out_dict[f"task_{i}.bias"] = getattr(self, f"task_{i}_bias")(features)"""
        out_dict["fc.weights"] = getattr(self, "fc_weights")(features)
        out_dict["fc.bias"] = getattr(self, "fc_bias")(features)
        return out_dict


class TargetNet(nn.Module):
    def __init__(self, hidden_size=256, out_dim=1,n_hiddens = 3, n_tasks=4):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.n_tasks = n_tasks

    def forward(self, x, weights=None):
        #print(weights[f'hidden{0}.weights'].reshape(self.hidden_size,21).unsqueeze(0).repeat((x.shape[0],1,1)).shape)
        # hidden layers
        x = F.linear(x, weight=weights[f'hidden{0}.weights'].reshape(self.hidden_size,14), 
                        bias=weights[f'hidden{0}.bias']) #256x1x256
        x = F.tanh(x)
        for i in range(1,self.n_hiddens):
            x = F.linear(x, weight=weights[f'hidden{i}.weights'].reshape(self.hidden_size,self.hidden_size),
                        bias=weights[f'hidden{i}.bias']) #256x1x256
            x = F.tanh(x)

        x = F.linear(x, weight=weights[f'fc.weights'].reshape(self.n_tasks,self.hidden_size),
                          bias=weights[f'fc.bias'])
        return x

