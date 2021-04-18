import json
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=4)
    return data


# T, B, 100 and T, B, N+1, 100 and T, B
def restricted_softmax(shared_layer, all_embed, qt,  mask):
    mask = mask.reshape(-1)
    shared_layer = shared_layer.reshape(-1,
                                        shared_layer.shape[2])[mask]  # TB, 100
    # TB, N+1, 100
    all_embed = all_embed.reshape(-1,
                                  all_embed.shape[2], all_embed.shape[3])[mask]
    qt = qt.reshape(-1, qt.shape[2])[mask]  # TB, N
    dot_prod = torch.sum(shared_layer.unsqueeze(1) *
                         all_embed, dim=-1)  # TB, N+1
    # TB and TB,N
    positive_prod, negative_prod = dot_prod[:, 0], dot_prod[:, 1:]
    with torch.no_grad():
        log_wk = (negative_prod.detach()-qt) - \
            torch.max(negative_prod.detach()-qt, dim=-1, keepdim=True)[0]
        wk = torch.exp(log_wk)
        wk = wk/torch.sum(wk, dim=-1, keepdim=True)
    loss = -positive_prod + torch.sum(negative_prod * wk, dim=-1)  # T, B
    return loss


# T, B, 100 and T, B, N+1, 100 and T, B
def restricted_sigmoid(shared_layer, all_embed, qt,  mask):
    mask = mask.reshape(-1)
    shared_layer = shared_layer.reshape(-1,
                                        shared_layer.shape[2])[mask]  # TB, 100
    # TB, N+1, 100
    all_embed = all_embed.reshape(-1,
                                  all_embed.shape[2], all_embed.shape[3])[mask]
    qt = qt.reshape(-1, qt.shape[2])[mask]  # TB, N
    dot_prod = torch.sum(shared_layer.unsqueeze(1) *
                         all_embed, dim=-1)  # TB, N+1
    label = torch.zeros(dot_prod.shape[0], dot_prod.shape[1]).to(device) + 1e-3
    label[:, 0] = 1.
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = (loss_fn(dot_prod, label)).mean(dim=-1)  # B, S
    return loss


def kl_(P, Q, L=None, eps=1e-6):  # L is list
    kl = P * ((P+eps) / (Q+eps)).log() + (1.-P) * \
        ((1.-P+eps)/(1.-Q+eps)).log()  # T, B, H
    if L is not None:
        mask = torch.zeros(P.shape[0], P.shape[1], 1).to(device)
        for idx, l in enumerate(L):
            mask[:l, idx, :] = 1.
        kl = kl * mask
    return kl
