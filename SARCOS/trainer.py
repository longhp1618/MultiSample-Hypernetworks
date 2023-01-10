from tqdm import trange
from collections import defaultdict
import torch
import numpy as np
import torch.nn.functional as F
from pymoo.factory import get_performance_indicator

import sys

from phn.solvers import MultiHead

import argparse
import json
from pathlib import Path

from tqdm import trange
from collections import defaultdict
import logging

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import (
    set_seed,
    set_logger,
    count_parameters,
    get_device,
    save_args,
)
from models import HyperNet, TargetNet

from load_data import get_data
from pymoo.factory import get_reference_directions

from phn.solvers import MultiHead
from scipy.spatial import Delaunay


def get_test_rays():
    """Create 100 rays for evaluation. Not pretty but does the trick"""
    """We follow the setting in paper Learning the Pareto Front with Hypernetwork"""
    test_rays = get_reference_directions("das-dennis", 7, n_partitions=11).astype(
        np.float32
    )
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    logging.info(f"initialize {len(test_rays)} test rays")
    return test_rays


@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device,epoch,name,n_tasks):
    hypernet.eval()
    results = defaultdict(list)
    loss_total = None
    #front = []
    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)

        ray /= ray.sum()

        total = 0.0
        full_losses = []
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            xs, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            pred = targetnet(xs, weights)

            # loss
            curr_losses = get_losses(pred, ys)

            ray = ray.squeeze(0)

            # losses
            full_losses.append(curr_losses.detach().cpu().numpy())
            total += bs
        if loss_total is None:
            loss_total = np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        else:
            loss_total += np.array(np.array(full_losses).mean(0).tolist(),dtype='float32')
        results["ray"].append(ray.cpu().numpy().tolist())
        results["loss"].append(np.array(full_losses).mean(0).tolist())
    print("\n")
    print(str(name)+" losses at "+str(epoch)+":",loss_total/len(rays))
    hv = get_performance_indicator(
        "hv",
        ref_point=np.ones(
            n_tasks,
        ),
    )
    hv_result = hv.do(np.array(results["loss"]))
    results["hv"] = hv_result

    return results



# ---------
# Task loss
# ---------
def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)

def train_MSH(solver,
    hnet,
    net,
    optimizer,
    optimizer_direction,
    device,
    n_mo_sol,
    n_tasks,
    head,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    test_rays,
    lamda,
    lr) -> None:
    #print(n_mo_sol)
    net_list = []
    for i in range(head):
        net_list.append(net)
    print("Start Phase 1")
    phase1_iter = trange(30)
    for _ in phase1_iter:
        penalty_epoch = []
        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer_direction.zero_grad()

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            losses_mean= []
            weights = []
            outputs = []
            rays = []
            penalty = []
            sum_penalty = 0

            for j in range(n_mo_sol):
                ray = np.random.dirichlet([1/n_tasks]*n_tasks, 1).astype(np.float32)
                rays.append(torch.from_numpy(
                                ray.flatten()
                            ).to(device))
                weights.append(hnet(rays[j]))
                #print(weights)
                outputs.append(net_list[j](x, weights[j]))
                losses_mean.append(torch.stack([F.mse_loss(outputs[j][:, k], y[:, k]) for k in range(n_tasks)
                                                ]))
                penalty.append(torch.sum(losses_mean[j]*rays[j])/(torch.norm(rays[j])*torch.norm(losses_mean[j])))
                sum_penalty += penalty[-1].item()
            penalty_epoch.append(sum_penalty/float(head))
            
            direction_loss = penalty[0]
            for phat in penalty[:]:
                direction_loss -= phat
            direction_loss.backward()
            optimizer_direction.step()
        print("Epochs {} penalty{:.2f}".format(_, np.mean(np.array(penalty_epoch))))
    print("End phase 1")

    print("Start Phase 2")
    epoch_iter = trange(epochs)
    count = 0
    patience = 0
    # early_stop = 0
    test_hv = -1
    val_hv = -1
    min_loss = [999]*n_tasks
    training_loss = []
    for epoch in epoch_iter:
        count += 1
        
        # if early_stop == 100:
        #     print("Early stop.")
        #     break
        
        if (patience+1) % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
            patience = 0
            lr *= np.sqrt(0.5)
        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            loss_torch = []
            loss_numpy = []
            weights = []
            outputs = []
            rays = []
            penalty = []

            sum_penalty = 0
            training_loss_epoch = []
            for j in range(n_mo_sol):       
                ray = np.random.dirichlet([1/n_tasks]*n_tasks, 1).astype(np.float32)

                ray = ray.astype(np.float32)
                rays.append(torch.from_numpy(
                                ray.flatten()
                            ).to(device))
                weights.append(hnet(rays[j]))
                outputs.append(net_list[j](x, weights[j]))
                loss_torch.append(torch.stack([F.mse_loss(outputs[j][:, k], y[:, k]) for k in range(n_tasks)]))
                loss_numpy.append([idx.cpu().detach().numpy() for idx in loss_torch[j]])
                penalty.append((torch.sum(loss_torch[j]*rays[j]))/(torch.norm(loss_torch[j])*
                                                torch.norm(rays[j])))
                sum_penalty += penalty[-1].item()

            loss_numpy = np.array(loss_numpy).T
            loss_numpy = loss_numpy[np.newaxis, :, :]
            training_loss_epoch.append(loss_numpy)
            total_dynamic_loss = solver.get_weighted_loss(loss_numpy,device,loss_torch,head,penalty,lamda)
            total_dynamic_loss.backward()
            optimizer.step()
        training_loss_epoch = np.mean(np.stack(training_loss_epoch), axis=0)
        training_loss.append(training_loss_epoch)
        for _ in range(n_tasks):
            min_loss[_] = min(min_loss[_], np.min(training_loss_epoch[0, _]))
        
        last_eval = epoch
        val_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=val_loader,
            rays=test_rays,
            device=device,
            name = "Val",
            epoch = epoch,
            n_tasks = n_tasks,
        )

        if val_epoch_results["hv"] > val_hv:
            val_hv = val_epoch_results["hv"]

            torch.save(hnet,'save_models/SARCOS_MH_Freely_'+ str(head) + "_" + str(lamda) + '_best.pt')

            patience = 0
        else:
            patience += 1
        print("Epoch", epoch, 'val_hv', val_hv)

    print("End Phase 2")

    torch.load('save_models/SARCOS_MH_Freely_'+ str(head) + "_" + str(lamda) + '_best.pt')

    test_epoch_results = evaluate(
        hypernet=hnet,
        targetnet=net,
        loader=test_loader,
        rays=test_rays,
        device=device,
        name = "Test",
        epoch = epoch,
        n_tasks = n_tasks,
    )
    pf = np.array(test_epoch_results["loss"])

    np.save("fronts/SARCOS_MH_Freely_"+ str(head) + "_" + str(lamda) + "_front.npy", pf)

    print("HV on test:",test_epoch_results["hv"])
    print("Best HV on val:",val_hv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARCOS")

    parser.add_argument("--datapath", type=str, default="data", help="path to data")
    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--seed", type=int, default=42, help="random seed")


    parser.add_argument("--n-mo-sol", type=int, default=8, help="random seed")

    parser.add_argument("--lamda", type=float, default=0.001, help="penalty parameter")

    args = parser.parse_args()
    

    set_seed(args.seed)
    set_logger()

    n_mo_obj = 7
    ref_point = [1]*n_mo_obj
    n_tasks = n_mo_obj
    head = args.n_mo_sol
    device=get_device(no_cuda=args.no_cuda, gpus=args.gpus)

    solver = MultiHead(args.n_mo_sol, n_mo_obj, ref_point)
    hnet: nn.Module = HyperNet()
    net: nn.Module = TargetNet()
    hnet = hnet.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=0.)
    optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=1e-4, weight_decay=0.)

    train_set, val_set, test_set = get_data(args.datapath)

    bs = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=0
    )


    test_rays = get_test_rays()
    lamda = args.lamda
    sys.stdout = open("log/Sarcos_MH_Freely_"+ str(head) + "_" + str(lamda) + "_log.txt", 'w')


    train_MSH(solver,
        hnet,
        net,
        optimizer,
        optimizer_direction,
        device,
        args.n_mo_sol,
        n_tasks,
        head,
        train_loader,
        val_loader,
        test_loader,
        args.n_epochs,
        test_rays,
        lamda,
        args.lr
        )

    # save_args(folder=args.out_dir, args=args)
    sys.stdout.close()