import os
import glob
import torch
import numpy as np
import torch.nn as nn
from dataset import data_generator
from torch.optim import Adam
import time
from utils import *
import argparse
from baseline import Baseline
from mnss import MNSS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, negative_mapper, generator, params):
    model.eval()
    total_loss = 0.
    batch_id = 0
    start_ = time.time()
    for (batch, negative_batch) in generator:
        if batch is None:
            print("Test Batch Dropped Due to Empty Skills/Majors!")
            continue
        with torch.no_grad():
            loss = model(batch, negative_batch)
            temp_loss = loss.detach().cpu().numpy()
            total_loss += temp_loss
        batch_id += 1
    return total_loss/batch_id


def train(model, negative_mapper, train_generator, test_generator, params):
    best_loss = None
    for epoch in range(params.n_epoch):
        start_ = time.time()
        model.train()
        total_loss = 0.
        batch_id = 0
        for (batch, negative_batch) in training_generator:
            if batch is None:
                print("Batch Dropped Due to Empty Skills/Majors!")
                continue
            if params.model in {'mnss'}:
                beta = min((params.max_beta/params.anneal) *
                           max(epoch-params.warmup, 0), params.max_beta)
                alpha = min((params.alpha/params.anneal) *
                            max(epoch-params.warmup, 0), params.alpha)
                loss = model(batch, negative_batch, beta=beta, alpha=alpha)
            else:
                loss = model(batch, negative_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            temp = loss.detach().cpu().numpy()
            total_loss += temp
            batch_id += 1
        total_loss /= batch_id
        test_loss = test(model, negative_mapper, test_generator, params)
        print("Epoch: {}, Train Loss: {}, Test Loss: {}".format(
            epoch, float(total_loss), float(test_loss)))
        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--negative_count', type=int, default=100,
                        help='')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='')
    parser.add_argument('--model', type=str, default='nemo',
                        help='')  # 'mnss', 'nss', 'nemo'
    parser.add_argument('--max_beta', type=float, default=0.1,
                        help='')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='')
    parser.add_argument('--warmup', type=int, default=10,
                        help='')
    parser.add_argument('--anneal', type=int, default=20,
                        help='')
    parser.add_argument('--pre_trained', action='store_true')
    parser.add_argument('--use_loc_ind', action='store_true')
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--dataset', type=str, default='demo')
    params = parser.parse_args()
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)

    data_dir = 'data/'
    ##
    negative_mapper = open_json(data_dir + 'demo_negative_mapper.json')
    mapper = open_json(data_dir + 'demo_mapper.json')
    data_path = data_dir + 'demo.json'
    num_workers = 2

    ####
    training_generator = data_generator(data_path, negative_count=params.negative_count, num_workers=num_workers,
                                        start_=0., end_=0.8, negative_mapper=negative_mapper, batch_size=params.batch_size,  drop_last=True, shuffle=True)

    testing_generator = data_generator(data_path, negative_count=params.negative_count, num_workers=num_workers, start_=0.8,
                                       end_=1., negative_mapper=negative_mapper, batch_size=params.test_batch_size, drop_last=True, shuffle=False)

    E = params.embed_dim
    embedding_dimensions = {'companies': E//2, 'locality': E//4, 'industry': E//4, 'degrees': E//4,
                            'schools': E//4, 'times': E//8, 'majors': E//4, 'intervals': E//8, 'occupations': E//2, 'skills': E}
    if params.pre_trained:
        embedding_dimensions['skills'] = 300
        embedding_dimensions['occupations'] = 300
        embedding_dimensions['companies'] = 300
    if params.model in {'nss', 'nemo'}:
        model = Baseline(mapper, embedding_dimensions, hidden_dim=params.hidden_dim, dropout=params.dropout,
                         pre_trained=params.pre_trained, use_loc_ind=params.use_loc_ind, data_dir=data_dir, model=params.model).to(device)
    if params.model in {'mnss'}:
        model = MNSS(mapper, embedding_dimensions, hidden_dim=params.hidden_dim,
                     dropout=params.dropout, pre_trained=params.pre_trained, use_loc_ind=params.use_loc_ind, data_dir=data_dir).to(device)

    optim = Adam(model.parameters(), lr=params.lr)
    # testing
    train(model, negative_mapper, training_generator, testing_generator, params)
