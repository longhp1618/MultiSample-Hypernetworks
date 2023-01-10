"""This code is referenced from code of paper
Learning the Pareto Front with Hypernetwork"""
"""https://github.com/AvivNavon/pareto-hypernetworks"""

"And paper Multi-Objective Learning to Predict Pareto Fronts Using Hypervolume Maximization"
"""https://github.com/timodeist/multi_objective_learning"""

import sys


import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import torch.nn.functional as F
from torch import nn
import logging
import argparse
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import pickle

from sklearn.model_selection import train_test_split

from data import Dataset

from tqdm import trange

from modules import functions_hv_grad_3d

from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv

from modules.functions_evaluation import fastNonDominatedSort
from modules.functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling

from modules.hv_maximization import HvMaximization

from model import LeNetHyper, LeNetTarget

import random

from utils import count_parameters, set_logger, set_seed, get_device, parse_arg

from evaluate import evaluate_hv
import timeit
from pathlib import Path
import settings as s 


def train(device: torch.device):
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  

  head = args.n_mo_sol
  lr = args.lr
  bs = args.batch_size
  epochs = args.epochs
  lamda = args.lamda

  training_loss = []
  val_hv = []
  val_loss = []
  best_hv = 0
  hnet: nn.Module = LeNetHyper([9, 5], ray_hidden_dim=args.ray_hidden)
  net: nn.Module = LeNetTarget([9, 5])
  logging.info(f"HN size: {count_parameters(hnet)}")

  hnet = hnet.to(device)
  net = net.to(device)

  loss1 = nn.CrossEntropyLoss(reduction='none')
  loss2 = nn.CrossEntropyLoss(reduction='none')
  loss1_mean = nn.CrossEntropyLoss()
  loss2_mean = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=args.wd)

  optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=1e-4, weight_decay=args.wd)

  assert args.val_size > 0, "please use validation by providing val_size > 0"
  data = Dataset(args.path, val_size=args.val_size)
  train_set, val_set, test_set = data.get_datasets()

  train_loader = torch.utils.data.DataLoader(
      dataset=train_set,
      batch_size=bs,
      shuffle=True,
      num_workers=4
  )
  val_loader = torch.utils.data.DataLoader(
      dataset=val_set,
      batch_size=bs,
      shuffle=True,
      num_workers=4
  )
  test_loader = torch.utils.data.DataLoader(
      dataset=test_set,
      batch_size=bs,
      shuffle=False,
      num_workers=4
  )

  ref_point = args.ref_point
  n_mo_sol = args.n_mo_sol
  n_mo_obj = args.n_mo_obj
  start = 0.1
  end = np.pi/2-0.1

  # Phase 1: warm up hypernet by optimizing only cosine similarity function
  print("Start Phase 1")
  phase1_iter = trange(1)
  for _ in phase1_iter:
    penalty_epoch = []
    for i, batch in enumerate(train_loader):
      hnet.train()
      optimizer_direction.zero_grad()

      img, ys = batch
      img = img.to(device)
      ys = ys.to(device)

      losses_mean= []
      weights = []
      outputs = []
      rays = []
      penalty = []
      sum_penalty = 0

      for j in range(n_mo_sol):
        random = np.random.uniform(start, end)
        ray = np.array([np.cos(random),
                          np.sin(random)], dtype='float32')

        ray /= np.sum(ray)

        ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2)) #one trick to help hypernet more general
                    
        rays.append(torch.from_numpy(
                        ray.flatten()
                    ).to(device))
        weights.append(hnet(rays[j]))

        outputs.append(torch.stack(net(img, weights[j]), dim=2))
        losses_mean.append(torch.stack([loss1_mean(outputs[j][:, :, 0], ys[:, 0]),
                                        loss2_mean(outputs[j][:, :, 1], ys[:, 1])]))
        penalty.append(torch.sum(losses_mean[j]*rays[j])/(torch.sqrt(torch.sum(rays[j]**2))*
                            torch.sqrt(torch.sum(losses_mean[j]**2))))
        sum_penalty += penalty[-1].item()
      penalty_epoch.append(sum_penalty/float(head))
    
      direction_loss = penalty[0]
      for phat in penalty[:]:
          direction_loss -= phat
      direction_loss.backward()
      optimizer_direction.step()
    print("Epochs {} penalty{:.2f}".format(_, np.mean(np.array(penalty_epoch))))
  print("End phase 1 after {}".format(_))

  print("Start Phase 2")
  epoch_iter = trange(epochs)
  mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)

  dem = 0

  patience = 0
  early_stop = 0

  min_loss_1 = 999
  min_loss_2 = 999

  for epoch in epoch_iter:
    dem += 1
    if early_stop == 35:
      print("Early stop.")
      break
      
    if (patience+1) % 10 == 0:
      for param_group in optimizer.param_groups:
          param_group['lr'] *= np.sqrt(0.5)
      patience = 0
      lr *= np.sqrt(0.5)
      print('Reduce the learning rate {}, penalty {:.2f}'.format(lr, lamda))

    for batch in train_loader:
      hnet.train()
      optimizer.zero_grad()
      img, ys = batch
      img = img.to(device)
      ys = ys.to(device)

      loss_torch_per_sample = []
      loss_numpy_per_sample = []
      loss_per_sample = []
      loss_torch = []
      loss_numpy = []
      weights = []
      outputs = []
      rays = []
      penalty = []

      sum_penalty = 0
      training_loss_epoch = []
      

      for j in range(n_mo_sol):
        if args.partition:
          random = np.random.uniform(start + j*(end-start)/head, start+ (j+1)*(end-start)/head) # partition sample
        else:
          random = np.random.uniform(start, end) # freely sample 

        ray = np.array([np.cos(random),
                          np.sin(random)], dtype='float32')
        
        ray /= ray.sum()
        ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2)) #some trick to make hypernetwork is more general

        rays.append(torch.from_numpy(
                        ray.flatten()
                    ).to(device))
        weights.append(hnet(rays[j]))

        outputs.append(torch.stack(net(img, weights[j]), dim=2))

        loss_per_sample = torch.stack([loss1(outputs[j][:, :, 0], ys[:, 0]), loss2(outputs[j][:, :, 1], ys[:, 1])])
        loss_torch_per_sample.append(loss_per_sample)
        loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())
        loss_mean = torch.mean(loss_per_sample, dim=1)
        loss_torch.append(loss_mean)
        loss_numpy.append(loss_mean.cpu().detach().numpy())
        penalty.append(torch.sum(loss_torch[j]*rays[j])/(torch.sqrt(torch.sum(loss_torch[j]**2))*
                                        torch.sqrt(torch.sum(rays[j]**2))))
        sum_penalty += penalty[-1].item()

      loss_numpy = np.array(loss_numpy).T
      loss_numpy = loss_numpy[np.newaxis, :, :]
      training_loss_epoch.append(loss_numpy)
      loss_numpy_per_sample = np.array(loss_numpy_per_sample).transpose(2, 1, 0)

      n_samples, n_mo_obj, n_mo_sol = loss_numpy.shape
      dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
      for i_sample in range(0, n_samples):
        weights_task = mo_opt.compute_weights(loss_numpy[i_sample,:,:])
        dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1,0)

      dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)

      i_mo_sol = 0

      total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                * loss_torch[i_mo_sol], dim=0))
      for i_mo_sol in range(1, head):
        total_dynamic_loss += torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :] 
                                            * loss_torch[i_mo_sol], dim=0))

      for phat in penalty:
        total_dynamic_loss -= lamda*phat

      total_dynamic_loss.backward()
      optimizer.step()

        
    training_loss_epoch = np.mean(np.stack(training_loss_epoch), axis=0)
    training_loss.append(training_loss_epoch)
    min_loss_1 = min(min_loss_1, np.min(training_loss_epoch[0, 0]))
    min_loss_2 = min(min_loss_2, np.min(training_loss_epoch[0, 1]))

    hnet.eval()
    hv, val_loss_epoch, val_min_penalty = evaluate_hv(args.n_rays, start, end, hnet, net, args.ref_point, val_loader, device)
    if hv > best_hv:
      best_hv = hv
      torch.save(hnet, Path(args.out_dir) / (args.data +"_MultiSample_" + args.part_text +  str(head) + "_" + str(lamda)+ "_.pt"))
      id = dem
      patience = 0
      print("Update Best Model at", id)
      early_stop = 0
    else:
      patience += 1
      early_stop += 1
    val_hv.append(hv)
    val_loss.append(val_loss_epoch)

    #print("hien thi kq")
    print("Epoch: {} val_hv: {:.2f} val_min_penalty: {:.2f} lr: {:.8f} penalty: {:.4f}".format(dem, hv, 
                                    val_min_penalty, lr, lamda))
    print("Training loss:", training_loss_epoch)
    print("Train Penalty:", [i.item() for i in penalty])
    print("best_val_hv: {:.4f} min_loss_1 {:.4f} min_loss_2 {:.4f} early_stop {}".format(best_hv, min_loss_1, min_loss_2, early_stop))
  
  output_dict = {"training_loss": training_loss,
                 "val_hv": val_hv,
                "val_loss": val_loss}

  save_results(args, test_loader, net, lamda, head, start, end)

  #return hnet, net, output_dict, id

def save_results(args, test_loader, net, lamda, head, start, end):
    hnet = torch.load(Path(args.out_dir) / (args.data + "_MultiSample_" + args.part_text +  str(head) + "_" + str(lamda) + ".pt"))
    results = []
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    random_lst = np.linspace(start, end, args.n_rays)

    for i in range(args.n_rays):
        random = random_lst[i]
        ray = np.array([np.cos(random),
                            np.sin(random)], dtype='float32')
        ray = torch.from_numpy(
                            ray.flatten()
                        ).to(device)

        weights = hnet(ray)
        loss_batch = []
        for i, batch in enumerate(test_loader):
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)
            output = torch.stack(net(img, weights), dim=2)
            loss_batch.append([loss1(output[:, :, 0], ys[:, 0]).cpu().detach().numpy(),
                            loss2(output[:, :, 1], ys[:, 1]).cpu().detach().numpy()])

        loss_batch = np.array(loss_batch)
        results.append(np.mean(loss_batch, axis=0))
    
    results = np.array(results, dtype='float32')
    np.save('fronts/' + args.data + "_MultiSample_" + args.part_text +  str(head) + "_" + str(lamda) + "_front.npy", results)
    print("HV testset:", compute_hv(results.T, args.ref_point))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST')
    parser.add_argument('--data_folder', type=str, default='data', help='path to data')

    parser.add_argument(
        "--data", type=str, choices=["MM", "MF", "FM"], default="MM", help="data name"
    )

    parser.add_argument('--epochs', type=int, default=400, help='num. epochs')
    parser.add_argument('--ray-hidden', type=int, default=100, help='lower range for ray')
    # parser.add_argument('--alpha', type=float, default=.2, help='alpha for dirichlet')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')

    parser.add_argument('--out-dir', type=str, default='save_models', help='outputs dir')
    parser.add_argument('--n-rays', type=int, default=25, help='num. testing rays')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()

    if args.data == "MM":
      settings = s.multi_mnist
    elif args.data == "MF":
      settings = s.multi_fashion
    else:
      settings = s.multi_fashion_mnist

    device=get_device(no_cuda=args.no_cuda, gpus=args.gpus)

    args.n_mo_sol = settings["n_mo_sol"]
    args.lamda = settings['lamda']
    args.ref_point = settings["ref_point"]
    args.n_mo_obj = settings['n_mo_obj']
    args.partition = settings['partition']

    # args.partition = True

    if args.partition:
      args.part_text = "partition_"
    else:
      args.part_text = "freely_"

    if args.data == "MM":
      args.path = args.data_folder + "/multi_mnist.pickle"
    elif args.data == "MF":
      args.path = args.data_folder + "/multi_fashion.pickle"
    else:
      args.path = args.data_folder + "/multi_fashion_and_mnist.pickle"

    set_seed(args.seed)
    set_logger()

    lamda = args.lamda
    head = args.n_mo_sol

    sys.stdout = open("log/" + args.data + "_MultiSample_" + args.part_text + "_" +  str(head) + str(lamda) + "_log.txt", 'w')
    start = timeit.default_timer()
    train(device = device)
    end = timeit.default_timer()

    print("Running time:",end-start)


sys.stdout.close()