import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset
from n_body_system.model import GNN, EGNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel
import os
from torch import nn, optim
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp-100-egnn_vel-3', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=210, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=3, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="_charged100_initvel1charged-100", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--gpu', type=int, default=0, help='GPU.')
parser.add_argument('--norm_type', type=str, default='scale_norm', 
                    help='available norm types: layer_norm, scale_norm, equi_norm')
parser.add_argument('--n_body', type=int, default=100, help='#num of bodies')
parser.add_argument('--delta', type=float, default=10, help='Translation distance')

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)

def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    dataset_train = NBodyDataset(partition='train', max_samples=args.max_training_samples, dataset_name=args.dataset)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.dataset)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.dataset)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)


    if args.model == 'gnn':
        model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model == 'egnn_vel':
        model = EGNN_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, norm_type=args.norm_type, n_body=args.n_body*args.batch_size)
    elif args.model == 'baseline':
        model = Baseline()
    elif args.model == 'linear_vel':
        model = Linear_dynamics(device=device)
    elif args.model == 'linear':
        model = Linear(6, 3, device=device)
    elif args.model == 'rf_vel':
        model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
    elif args.model == 'se3_transformer' or args.model == 'tfn':
        from n_body_system.se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        model = SE3_Transformer(n_particles=5, n_dimesnion=3, nf=int(args.nf/args.degree), n_layers=args.n_layers, model=args.model, num_degrees=args.degree, div=1)
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        raise Exception("Wrong model specified")
    
    if args.norm_type != 'None':
        model = torch.load('n_body_system/models/' + args.exp_name + args.norm_type + '1.pt')
    else:
        model = torch.load('n_body_system/models/' + args.exp_name + '.pt')
    model = model.to(device)

    print(model)

    train_loc_pred, train_loc_end = train(model, loader_train, args.delta, backprop=False)
    train_loc_pred1 = train_loc_end + args.delta
    val_loc_pred, val_loc_end = train(model, loader_val, args.delta, backprop=False)
    val_loc_pred1 = val_loc_end + args.delta
    test_loc_pred, test_loc_end = train(model, loader_test, args.delta, backprop=False)
    test_loc_pred1 = test_loc_end + args.delta

    print (args.exp_name, args.norm_type, args.delta)
    print ('Train:', round(mean_squared_error(train_loc_pred, train_loc_pred1), 4))
    print ('Val:', round(mean_squared_error(val_loc_pred, val_loc_pred1), 4))
    print ('Test:', round(mean_squared_error(test_loc_pred, test_loc_pred1), 4))
    # print ('Train:', mean_squared_error(train_loc_pred, train_loc_pred1))
    # print ('Val:', mean_squared_error(val_loc_pred, val_loc_pred1))
    # print ('Test:', mean_squared_error(test_loc_pred, test_loc_pred1))
    
    return train_loc_pred, val_loc_pred, test_loc_pred


def train(model, loader, delta, backprop=False):
    if backprop:
        model.train()
    else:
        model.eval()
    
    res, res1 = [], []

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data
        loc1 = loc + torch.ones(size=loc.size()).to(device) * delta

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]


        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'gnn':
            nodes = torch.cat([loc, vel], dim=1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model == 'egnn':
            nodes = torch.ones(loc.size(0), 1).to(device)  # all input nodes are set to 1
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            vel_attr = get_velocity_attr(loc, vel, rows, cols).detach()
            edge_attr = torch.cat([edge_attr, loc_dist, vel_attr], 1).detach()  # concatenate all edge properties

            loc_pred = model(nodes, loc.detach(), edges, edge_attr)
        elif args.model == 'egnn_vel':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            # loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
            loc_pred1 = model(nodes, loc1.detach(), edges, vel, edge_attr)
        elif args.model == 'baseline':
            backprop = False
            loc_pred = model(loc)
        elif args.model == 'linear':
            loc_pred = model(torch.cat([loc, vel], dim=1))
        elif args.model == 'linear_vel':
            loc_pred = model(loc, vel)
        elif args.model == 'se3_transformer' or args.model == 'tfn':
            loc_pred = model(loc, vel, charges)
        elif args.model == 'rf_vel':
            rows, cols = edges
            vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1).unsqueeze(1)).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred = model(vel_norm, loc.detach(), edges, vel, edge_attr)
        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
          
        res = res + loc_pred1.cpu().detach().numpy().tolist()
        res1 = res1 + loc_end.cpu().detach().numpy().tolist()

    return np.array(res), np.array(res1)


def main_sweep():
    training_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000, 50000]
    n_epochs = [2000, 2000, 4000, 5000, 8000, 10000, 8000, 6000, 4000, 2000]
    if args.model == 'egnn_vel':
        n_epochs = [4000, 4000, 2000, 2000, 2000, 1500, 1500, 1500, 1000, 1000] # up to the 5th updated
    elif args.model == 'kholer_vel':
        n_epochs = [8000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000, 2000] # up to the 5th

    if args.sweep_training == 2:
        training_samples = training_samples[0:5]
        n_epochs = n_epochs[0:5]
    elif args.sweep_training == 3:
        training_samples = training_samples[6:]
        n_epochs = n_epochs[6:]
    elif args.sweep_training == 4:
        training_samples = training_samples[8:]
        n_epochs = n_epochs[8:]


    results = {'tr_samples': [], 'test_loss': [], 'best_epochs': []}
    for epochs, tr_samples in zip(n_epochs, training_samples):
        args.epochs = epochs
        args.max_training_samples = tr_samples
        args.test_interval = max(int(10000/tr_samples), 1)
        best_val_loss, best_test_loss, best_epoch = main()
        results['tr_samples'].append(tr_samples)
        results['best_epochs'].append(best_epoch)
        results['test_loss'].append(best_test_loss)
        print("\n####### Results #######")
        print(results)
        print("Results for %d epochs and %d # training samples \n" % (epochs, tr_samples))


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)
    if args.sweep_training:
        main_sweep()
    else:
        main()




