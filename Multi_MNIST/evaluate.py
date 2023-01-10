import numpy as np
from torch import nn
import torch
from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv
def evaluate_hv(soluong, start, end, hnet, net, ref_point, data_loader, device):
  results = []
  loss1 = nn.CrossEntropyLoss()
  loss2 = nn.CrossEntropyLoss()
  angles = np.linspace(start, end, soluong, endpoint=True)
  x = np.cos(angles)
  y = np.sin(angles)
  rays = np.c_[x, y]
  min_penalty= 999
  for i in range(soluong):
    ray = rays[i]
    ray /= ray.sum()
    ray = torch.from_numpy(
                        ray.astype(np.float32).flatten()
                    ).to(device)

    weights = hnet(ray)
    loss_batch = []
    for _, batch in enumerate(data_loader):
      img, ys = batch
      img = img.to(device)
      ys = ys.to(device)
      output = torch.stack(net(img, weights), dim=2)
      l1 = loss1(output[:, :, 0], ys[:, 0]).cpu().detach().numpy()
      l2 = loss2(output[:, :, 1], ys[:, 1]).cpu().detach().numpy()
      loss_batch.append([l1, l2])
    loss_batch = np.array(loss_batch)
    loss_mean = np.mean(loss_batch, axis=0)
    penalty = np.sum(rays[i].astype(np.float32).flatten()*loss_mean)/(np.sqrt(np.sum(loss_mean**2)))
    if penalty < min_penalty:
      min_penalty = penalty
    results.append(loss_mean)
      
  results = np.array(results, dtype='float32')
  return compute_hv(results.T, ref_point), results, min_penalty