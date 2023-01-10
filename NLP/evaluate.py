from torch import nn
import torch.nn
import numpy as np
from pymoo.factory import get_performance_indicator

def evaluate_hv(soluong, start, end, hnet, net, val_loader, embedding_matrix, device):
  #print('Evaluate')
  results = []
  loss2 = nn.CrossEntropyLoss()
  loss1 = nn.MSELoss()
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
    for _, batch in enumerate(val_loader):
      x_batch, y_batch = batch
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      logit1, logit2 = net(x_batch, weights, embedding_matrix)
      l1 = loss1(logit1.squeeze(1), y_batch[:, 0]).cpu().detach().numpy()
      l2 = loss2(logit2.squeeze(1), y_batch[:, 1].long()).cpu().detach().numpy()
      loss_batch.append([l1, l2])
    loss_batch = np.array(loss_batch)
    loss_mean = np.mean(loss_batch, axis=0)
    penalty = np.sum(rays[i].astype(np.float32).flatten()*loss_mean)/(np.sqrt(np.sum(loss_mean**2)))
    if penalty < min_penalty:
      min_penalty = penalty
    results.append(loss_mean)
      
  hv = get_performance_indicator(
      "hv",
      ref_point=2*np.ones(
          2,
      ),
  )
  hv_result = hv.do(np.array(results, dtype='float32'))
  return hv_result, np.array(results, dtype='float32'), min_penalty
