from __future__ import print_function, division
import random
from tqdm import tqdm
from util.clusteringPerformance import StatisticClustering
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from model import DBONet
from util.utils import features_to_adj, normalization, standardization
import os

def train(args,features, adj,  labels, n_view, n_clusters, model, optimizer, scheduler, device):
    acc_max = 0.0
    res = []
    features_norm={}
    # data tensor
    for i in range(n_view):
        features_norm[i]= torch.from_numpy(features[i]/1.0).float().to(device)
        features_norm[i]= standardization(normalization(features_norm[i]))
        features[i]= torch.Tensor(features[i] / 1.0).to(device)
        adj[i]=adj[i].to_dense().float().to(device)

    criterion = nn.MSELoss()
    with tqdm(total=args.epoch, desc="Training") as pbar:
        for i in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            output_z = model(features, adj)


            loss_dis = torch.Tensor(np.array([0])).to(device)
            loss_lap = torch.Tensor(np.array([0])).to(device)
            for k in range(n_view):
                loss_dis += criterion(output_z.mm(output_z.t()), features_norm[k].mm(features_norm[k].t()))
                loss_lap += criterion(output_z.mm(output_z.t()), adj[k])

            loss = loss_dis + args.gamma * loss_lap
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            Dis_loss = loss_dis.cpu().detach().numpy()
            Lap_loss = loss_lap.cpu().detach().numpy()
            train_loss = loss.cpu().detach().numpy()
            output_zz = output_z.detach().cpu().numpy()

            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_zz, labels, n_clusters)
            if (ACC[0] > acc_max):
                acc_max = ACC[0]
                res = []
                for item in [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]:
                    res.append("{}({})".format(item[0] * 100, item[1] * 100))
            pbar.update(1)
            print({"Dis_loss": "{:.6f}".format(Dis_loss[0]), "Lap_loss": "{:.6f}".format(Lap_loss[0]),
                   'Loss': '{:.6f}'.format(train_loss[0])})
    return res


def getInitF(dataset, n_view, dataset_dir):
    dataset=dataset + "WG"
    data = scipy.io.loadmat(os.path.join(dataset_dir, dataset))
    Z = data[dataset]
    Z_init = Z[0][0]
    for i in range(1, Z.shape[1]):
        Z_init += Z[0][i]
    return Z_init / n_view


def main(data, args):
    # Clustering evaluation metrics
    SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    adj, features, labels, nfeats, n_view, n_clusters = features_to_adj(data, args.path + args.data_path)


    n = len(adj[0])
    print(f"samples:{n}, view size:{n_view}, feature dimensions:{nfeats}, class:{n_clusters}")

    # initial representation
    Z_init = getInitF(data , n_view, args.path + args.data_path)

    # network architecture
    model = DBONet(nfeats, n_view, n_clusters, args.block, args.thre,  Z_init, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, verbose=True,
                                                           min_lr=1e-8)

    print(f"gamma:{args.gamma}, block:{args.block}, epoch:{args.epoch}, thre:{args.thre}, lr:{args.lr}")

    # Training
    res = train(args, features, adj,  labels, n_view, n_clusters, model, optimizer, scheduler, device)


    print("{}:{}\n".format(data, dict(zip(SCORE, res))))
