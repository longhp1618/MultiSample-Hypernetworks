"""This code is referenced from code of paper
Learning the Pareto Front with Hypernetwork"""
"""https://github.com/AvivNavon/pareto-hypernetworks"""

"And paper Multi-Objective Learning to Predict Pareto Fronts Using Hypervolume Maximization"
"""https://github.com/timodeist/multi_objective_learning"""

import torch
import torch.nn.functional as F
from torch import nn
import logging
import argparse
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import pickle

from sklearn.model_selection import train_test_split

from data import dataset

from tqdm import trange

from modules import functions_hv_grad_3d

from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv

from modules.functions_evaluation import fastNonDominatedSort
from modules.functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling

from modules.hv_maximization import HvMaximization

from model import CNN_Target, CNN_Hyper

import random

from utils import count_parameters, set_logger, set_seed, get_device, parse_arg

from evaluate import evaluate_hv

import timeit
from pathlib import Path
import time
import sys

from settings import NLP


def train(args, lamda, device):
  
  ref_point = args.ref_point
  n_mo_sol = args.n_mo_sol
  n_mo_obj = args.n_mo_obj

  head = n_mo_sol

  train_loader, valid_loader, test_loader, emb = dataset(args.data_folder)
  emb = torch.tensor(emb, dtype=torch.float32).to(device)
  emb.requires_grad = False

  hnet: nn.Module = CNN_Hyper(args.embed_size, args.num_filters, args.ray_hidden_dim)
  net: nn.Module = CNN_Target(args.embed_size, args.num_filters)


  logging.info(f"HN size: {count_parameters(hnet)}")
  hnet = hnet.to(device)

  net = net.to(device)
  
  optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr)
  num_ray = args.n_rays

  loss1 = nn.MSELoss()
  loss2 = nn.CrossEntropyLoss()

  val_min_penalty = 0.
  best_hv = -1
  
  start_time = time.time()
  print("Start Phase 1")
  optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=args.lr)
  phase1_iter = trange(1)

  start = 0.1
  end = np.pi/2-0.1

  for _ in phase1_iter:
    penalty_epoch = []
    for i, batch in enumerate(train_loader):
      hnet.train()
      optimizer_direction.zero_grad()

      x_batch, y_batch = batch
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      losses_mean= []
      weights = []

      rays = []
      penalty = []
      sum_penalty = 0

      for j in range(n_mo_sol):
        random = np.random.uniform(start, end)
        ray = np.array([np.cos(random),
                          np.sin(random)], dtype='float32')
        ray /= np.sum(ray)
        ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2)) #a trick to make hypernetwork is more general
                    
        rays.append(torch.from_numpy(
                        ray.flatten()
                    ).to(device))
        weights.append(hnet(rays[j]))

        logit1, logit2 = net(x_batch, weights[j], emb)
        losses_mean.append(torch.stack([loss1(logit1.squeeze(1), y_batch[:, 0]),
                                        loss2(logit2.squeeze(1), y_batch[:, 1].long())]))
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

  print("Start Phase 2.")
  epoch_iter = trange(args.epochs)
  mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)

  training_loss = []
  val_hv = []
  val_loss = []
  patience = 0
  early_stop = 0
  min_loss_1 = 999
  min_loss_2 = 999

  for epoch in epoch_iter:
      if early_stop == 35:
          break
      if (patience+1) % 10 == 0:
          for param_group in optimizer.param_groups:
              param_group['lr'] *= np.sqrt(0.5)
              patience = 0
          lr = param_group['lr']
          print("Reduce Learning rate", param_group['lr'], "at", epoch)

      hnet.train()
      for i, (x_batch, y_batch) in enumerate(train_loader):
          # Predict/Forward Pass
          optimizer.zero_grad()
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)

          loss_torch = []
          loss_numpy = []
          weights = []
          outputs = []
          rays = []
          penalty = []
          sum_penalty = 0.
          training_loss_epoch = []
          for j in range(n_mo_sol):
            if args.partition:
              random = np.random.uniform(start + j*(end-start)/head, start+ (j+1)*(end-start)/head)
            else:
              random = np.random.uniform(start, end)

            ray = np.array([np.cos(random),
                              np.sin(random)], dtype='float32')
            
            ray /= ray.sum()
            ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2)) #some trick to make hypernetwork is more general

            rays.append(torch.from_numpy(
                            ray.flatten()
                        ).to(device))
            weights.append(hnet(rays[j]))
            logit1, logit2 = net(x_batch, weights[j], emb)
            loss_mean = torch.stack((loss1(logit1.squeeze(1), y_batch[:, 0]), loss2(logit2.squeeze(1), y_batch[:, 1].long())))
            loss_torch.append(loss_mean)
            loss_numpy.append(loss_mean.cpu().detach().numpy())
            penalty.append(torch.sum(loss_torch[j]*rays[j])/(torch.sqrt(torch.sum(loss_torch[j]**2))*
                                        torch.sqrt(torch.sum(rays[j]**2))))
            sum_penalty += penalty[-1].item()

          loss_numpy = np.array(loss_numpy).T
          loss_numpy = loss_numpy[np.newaxis, :, :]
          training_loss_epoch.append(loss_numpy)


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
      hv, val_loss_epoch, val_min_penalty = evaluate_hv(num_ray, start, end, hnet, net, valid_loader, emb, device)
      if best_hv < hv:
          best_hv = hv
          torch.save(hnet, 'save_models/NLP_MultiSample_' + args.part_text + str(head) + '_' + str(lamda) +".pt")
          print("Update best model at", epoch)
          patience = 0
          early_stop = 0
      else:
          early_stop += 1
          patience += 1

      val_hv.append(hv)
      val_loss.append(val_loss_epoch)

      print("Epoch: {} val_hv: {:.2f} val_min_penalty: {:.2f} penalty: {:.4f}".format(epoch, hv, 
                                      val_min_penalty, lamda))
      print("Training loss:", training_loss_epoch)
      print("Train Penalty:", [i.item() for i in penalty])
      print("Best_val_hv: {:.4f} min_loss_1 {:.4f} min_loss_2 {:.4f} early_stop {}".format(
        best_hv, min_loss_1, min_loss_2, early_stop))

  output_dict = {"training_loss": training_loss,
                 "val_hv": val_hv,
                "val_loss": val_loss}


  elapsed_time = time.time() - start_time 

  hnet = torch.load('save_models/NLP_MultiSample_' + args.part_text + str(head) + '_' + str(lamda)+".pt")

  hv_epoch, test_loss, test_min_penalty = evaluate_hv(25, start, end, hnet, net, test_loader, emb, device)
  test_loss = np.array(test_loss)
  np.save('fronts/NLP_MultiSample_' + args.part_text + str(head) + '_' + str(lamda) + "_front.npy", test_loss)
  print("HV:", hv_epoch)

  np.save('front/NLP_MultiSample_' + args.part_text + str(head) + '_' + str(lamda)+".npy", val_loss)
  return hnet, net, elapsed_time, output_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiMNIST')
    parser.add_argument('--data-folder', type=str, default='input/', help='path to data')

    parser.add_argument('--epochs', type=int, default=500, help='num. epochs')
    parser.add_argument('--ray-hidden', type=int, default=100, help='lower range for ray')

    parser.add_argument('--gpus', type=str, default='0', help='gpu device')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')

    parser.add_argument('--out-dir', type=str, default='outputs', help='outputs dir')
    parser.add_argument('--n-rays', type=int, default=25, help='num. testing rays')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-filters', type=int, default=36, help='number of filters of Conv2D in target network')
    parser.add_argument('--embed-size', type=int, default=300, help='embedding size')
    parser.add_argument('--ray-hidden-dim', type=int, default=100, help='hidden dim of hypernetwork')

    args = parser.parse_args()
    
    
    args.ref_point = NLP['ref_point']
    args.n_mo_obj = NLP['n_mo_obj']
    args.n_mo_sol = NLP['n_mo_sol']
    args.partition = NLP['partition']
    args.lamda = NLP['lamda']
    if args.partition:
      args.part_text = "partition_"
    else:
      args.part_text = "freely_"

    head = args.n_mo_sol
    lamda = args.lamda

    sys.stdout = open('log/NLP_MultiSample_' + args.part_text + str(head) + '_' + str(lamda) + '.txt', 'w')
    
    device = get_device(gpus=args.gpus)
    
    set_logger()
    set_seed(args.seed)
    hnet, net, elapsed_time, outdict = train(args, lamda, device)

    print("Running time:", elapsed_time)
    sys.stdout.close()
