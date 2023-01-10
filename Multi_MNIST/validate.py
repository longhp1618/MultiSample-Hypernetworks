import logging
import argparse
import json
from collections import defaultdict
from opcode import HAVE_ARGUMENT
from os import access
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

from data import Dataset
from model import LeNetHyper, LeNetTarget
from utils import (get_device,set_seed, circle_points)

from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv
import random
@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    results = defaultdict(list)

    for ray in rays:
        total = 0.
        task1_correct, task2_correct = 0., 0.
        l1, l2 = 0., 0.
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()

        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            img, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            logit1, logit2 = targetnet(img, weights)

            # loss
            curr_l1 = loss1(logit1, ys[:, 0])
            curr_l2 = loss2(logit2, ys[:, 1])
            l1 += curr_l1 * bs
            l2 += curr_l2 * bs

            # acc
            pred1 = logit1.data.max(1)[1]  # first column has actual prob.
            pred2 = logit2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()

            total += bs

        results['ray'].append(ray.squeeze(0).cpu().numpy().tolist())
        results['task1_acc'].append(task1_correct.cpu().item() / total)
        results['task2_acc'].append(task2_correct.cpu().item() / total)
        results['task1_loss'].append(l1.cpu().item() / total)
        results['task2_loss'].append(l2.cpu().item() / total)
    losses = np.stack((results['task1_loss'], results['task2_loss']))
    acc = np.stack((results['task1_acc'], results['task2_acc']))
    return results, compute_hv(losses, (2, 2)), losses, acc
def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(1)
path = "/home/ubuntu/long.hp/Multi_MNIST/Clean_code/data/multi_fashion.pickle"
val_size = 0.1
data = Dataset(path, val_size=val_size)
bs = 256
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

device=get_device(no_cuda=False, gpus='0')


net: nn.Module = LeNetTarget([9, 5])
net = net.to(device)

n_rays = 25
min_angle = 0.1
max_angle = np.pi/2-0.1


test_rays = circle_points(n_rays, min_angle=min_angle, max_angle=max_angle)

# head = 8
# hesophat = 5.0
#hnet = torch.load("/home/ubuntu/long.hp/Multi_MNIST/Clean_code/outputs/"+"MM_" + str(head) + "_" + str(hesophat)+ "_.pt")
hnet = torch.load("/home/ubuntu/long.hp/Multi_MNIST/Clean_code/save_models/MF_MultiSample_freely__164.0_.pt", map_location=device)
hnet = hnet.to(device)
_, hv, pf, acc = evaluate(hnet, net, test_loader, test_rays, device)
print("hv:", hv)
# np.save("outputs/MM_" + str(head) + "_" + str(hesophat)+ "_front.npy", pf)
# np.save("outputs/MM_" + str(head) + "_" + str(hesophat)+ "_ac.npy", acc)

np.save("fronts/MF_MultiSample_freely__164.0_.npy", pf)
#np.save("outputs/MFM_non_adap_16_4.0_ac.npy", acc)