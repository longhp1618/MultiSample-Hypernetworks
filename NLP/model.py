import torch
import torch.nn.functional as F
from torch import nn

class CNN_Hyper(nn.Module):
  def __init__(self, embed_size=300, num_filters=36, ray_hidden_dim=100):
    super(CNN_Hyper, self).__init__()
    self.filter_sizes = [1,2,3,5]
    self.num_filters = num_filters
    self.embed_size = embed_size

    self.n_classes = 100
    self.dropout = nn.Dropout(0.1)

    self.ray_mlp = nn.Sequential(
        nn.Linear(2, ray_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(ray_hidden_dim, ray_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(ray_hidden_dim, ray_hidden_dim)
    )

    for i in range(len(self.filter_sizes)):
      setattr(self, f"conv{i}.weights", nn.Linear(ray_hidden_dim, self.num_filters*self.filter_sizes[i]*self.embed_size))
      setattr(self, f"conv{i}.bias", nn.Linear(ray_hidden_dim, self.num_filters))
    
    setattr(self, f"MSE.weights", nn.Linear(ray_hidden_dim, len(self.filter_sizes)*self.num_filters))
    setattr(self, f"MSE.bias", nn.Linear(ray_hidden_dim, 1))
    setattr(self, f"CE.weights", nn.Linear(ray_hidden_dim, self.n_classes*len(self.filter_sizes)*self.num_filters))
    setattr(self, f"CE.bias", nn.Linear(ray_hidden_dim, self.n_classes))
  def forward(self, ray):
    features = self.ray_mlp(ray)
    out_dict = {}
    for i in range(len(self.filter_sizes)):
      out_dict[f"conv{i}.weights"] = self.dropout(getattr(self, f"conv{i}.weights")(features))
      out_dict[f"conv{i}.bias"] = self.dropout(getattr(self, f"conv{i}.bias")(features).flatten())
    out_dict[f"MSE.weights"] = self.dropout(getattr(self, f"MSE.weights")(features))
    out_dict[f"MSE.bias"] = getattr(self, f"MSE.bias")(features).flatten()
    out_dict[f"CE.weights"] = self.dropout(getattr(self, f"CE.weights")(features))
    out_dict[f"CE.bias"] = getattr(self, f"CE.bias")(features).flatten()
    return out_dict


class CNN_Target(nn.Module):
    
    def __init__(self, embed_size=300, num_filters=36):
        super(CNN_Target, self).__init__()
        self.filter_sizes = [1,2,3,5]
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.n_classes = 100
        #n_classes = len(le.classes_)
        #self.embedding = nn.Embedding(max_features, embed_size)
        #self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        #self.embedding.weight.requires_grad = False
        #self.convs1 = nn.ModuleList([F.conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        #self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)


    def forward(self, x, weights, embedding_matrix):
        x = F.embedding(x, embedding_matrix)  
        x = x.unsqueeze(1)
        x_lst = []
        for i in range(len(self.filter_sizes)):
          x_lst.append(F.relu(F.conv2d(x, weight=weights[f'conv{i}.weights'].reshape(self.num_filters, 1, self.filter_sizes[i], self.embed_size),
                              bias=weights[f'conv{i}.bias'])).squeeze(3))
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_lst]  
        x = torch.cat(x, 1)
        #x = self.dropout(x)  
        logit_MSE = F.linear(x, weight=weights[f'MSE.weights'].reshape(1, len(self.filter_sizes)*self.num_filters),
                         bias=weights[f'MSE.bias'])
        logit_CE = F.linear(x, weight=weights[f'CE.weights'].reshape(self.n_classes, len(self.filter_sizes)*self.num_filters),
                         bias=weights[f'CE.bias'])
        logits = [logit_MSE, logit_CE]
        return logits