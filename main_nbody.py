import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset, MD22Dataset
from n_body_system.model import GNN, EGNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, ClofNet_vel
import os
from torch import nn, optim
import json
import time

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--task', type=str, default="N_body_system", metavar='N',
                    help='determine which task to do')

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

parser.add_argument('--n_body', type=int, default=5, help='#num of bodies')
parser.add_argument('--local_path', type=str, default='/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/', help='data loader and model save directory path')

# MD dataset preprocessing
parser.add_argument('--delta_frame', type=int, default=100, help='#number of time steps interval')
parser.add_argument('--cut_off', action="store_true", help='whether apply local distance cutoff')
parser.add_argument('--h_atoms', type=eval, default=True, help='whether removing hydrogen atoms')
parser.add_argument('--cutoff_lambda', type=float, default=20, help='cutoff distance threshold')

# norm and sc
parser.add_argument('--norm_type', type=str, default='None',
                    help='available norm types: layer_norm, scale_norm, equi_norm')
parser.add_argument('--local_frame', type=str, default='PCA',
                    help='local frame construction methods: PCA, GSP, CP')
parser.add_argument('--sigma', type=str, default='std',
                    help='scaling factor: std, min-max, norm')

# testing E(3) equivariance
parser.add_argument("--test_rot", action="store_true", help="Rotate the test")
parser.add_argument("--test_trans", action="store_true", help="Translate the test")
parser.add_argument("--test_ref", action="store_true", help="Reflect the test")


time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

print(args.test_trans)

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass


if args.norm_type != 'equi_norm':
    file_name = args.outf + "/" + args.task + '-' + args.dataset + '-' + args.model + '-' + str(args.norm_diff) + '-' + args.norm_type + '-' + str(args.n_body) + '-' + \
                str(args.seed) + '-' + str(args.n_layers) + '-' + str(args.lr) + '-' + str(args.cutoff_lambda)
else:
    file_name = args.outf + "/" + args.task + '-' + args.dataset + '-' + args.model + '_' + str(args.norm_diff) + '-' + args.norm_type + '-' + str(args.n_body) + '-' + \
                str(args.seed) + '-' + str(args.n_layers) + '-' + str(args.lr) + '-' + str(args.cutoff_lambda) + '-' + \
                args.sigma + '-' + args.local_frame

try:
    os.makedirs(file_name)
except OSError:
    pass

def get_velocity_attr(loc, vel, rows, cols):
    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # N-body system task
    if args.task == 'N_body_system':
        dataset_train = NBodyDataset(partition='train', max_samples=args.max_training_samples, dataset_name=args.dataset,
                                     cutoff=args.cut_off, cutoff_lambda=args.cutoff_lambda)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        dataset_val = NBodyDataset(partition='val', dataset_name=args.dataset, cutoff=args.cut_off,
                                   cutoff_lambda=args.cutoff_lambda)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

        dataset_test = NBodyDataset(partition='test', dataset_name=args.dataset, test_rot=args.test_rot,
                                    test_trans=args.test_trans, test_ref=args.test_ref,
                                    cutoff=args.cut_off, cutoff_lambda=args.cutoff_lambda)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # MD22 task
    elif args.task == 'MD22':
        dataset_train = MD22Dataset(partition='train', max_samples=args.max_training_samples, delta_frame=args.delta_frame,
                                    data_dir=args.local_path+'MD22', molecule_type=args.dataset, cutoff = args.cut_off,
                                    hydrogen_atoms=args.h_atoms, cutoff_lambda=args.cutoff_lambda)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

        dataset_val = MD22Dataset(partition='val', max_samples=args.max_training_samples, delta_frame=args.delta_frame,
                                  data_dir=args.local_path+'MD22', molecule_type=args.dataset, cutoff = args.cut_off,
                                  hydrogen_atoms=args.h_atoms, cutoff_lambda=args.cutoff_lambda)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 drop_last=False)
        dataset_test = MD22Dataset(partition='test', max_samples=args.max_training_samples, delta_frame=args.delta_frame,
                                  data_dir=args.local_path+'MD22', molecule_type=args.dataset, cutoff = args.cut_off,
                                   hydrogen_atoms=args.h_atoms, test_rot=args.test_rot, test_trans=args.test_trans,
                                   test_ref = args.test_ref, cutoff_lambda=args.cutoff_lambda)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False)

    # chignolin task
    elif args.task == 'Chignolin':
        dataset_train = ChignolinDataset(partition='train', max_samples=args.max_training_samples,
                                     dataset_name=args.dataset)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

        dataset_val = ChignolinDataset(partition='val', dataset_name=args.dataset)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 drop_last=False)

        dataset_test = ChignolinDataset(partition='test', dataset_name=args.dataset)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False)

    if args.model == 'gnn':
        model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model == 'egnn_vel':
        if args.task == 'N_body_system':
            model = EGNN_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                             batch_size=args.batch_size, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,
                             norm_type=args.norm_type, n_body=args.n_body*args.batch_size, sigma=args.sigma,
                             local_frame=args.local_frame)
        elif args.task == 'MD22':
            n_body_size = dataset_train[0][0].shape[0]
            model = EGNN_vel(in_node_nf=1, in_edge_nf=4, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                             batch_size=args.batch_size, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,
                             norm_type=args.norm_type, n_body=n_body_size*args.batch_size, sigma=args.sigma,
                             local_frame=args.local_frame)
    elif args.model == 'baseline':
        model = Baseline()
    elif args.model == 'linear_vel':
        model = Linear_dynamics(device=device)
    elif args.model == 'linear':
        model = Linear(6, 3, device=device)
    elif args.model == 'rf_vel':
        if args.task == 'N_body_system':
            model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2, device=device, act_fn=nn.SiLU(), batch_size=args.batch_size,
                           n_layers=args.n_layers, norm_type=args.norm_type, n_body=args.n_body*args.batch_size,
                           sigma=args.sigma, local_frame=args.local_frame)
        elif args.task == 'MD22':
            n_body_size = dataset_train[0][0].shape[0]
            model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2, device=device, act_fn=nn.SiLU(),
                           batch_size=args.batch_size,
                           n_layers=args.n_layers, norm_type=args.norm_type, n_body=n_body_size * args.batch_size,
                           sigma=args.sigma, local_frame=args.local_frame)
    elif args.model == 'clof_vel':
        if args.task == 'N_body_system':
            model = ClofNet_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, n_layers=args.n_layers, device=device,
                                batch_size=args.batch_size, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,
                                norm_type=args.norm_type, n_body=args.n_body*args.batch_size, sigma=args.sigma,
                                local_frame=args.local_frame)
        elif args.task == 'MD22':
            n_body_size = dataset_train[0][0].shape[0]
            model = ClofNet_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, n_layers=args.n_layers, device=device,
                                batch_size=args.batch_size, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,
                                norm_type=args.norm_type, n_body=n_body_size*args.batch_size, sigma=args.sigma,
                                local_frame=args.local_frame)
    elif args.model == 'se3_transformer' or args.model == 'tfn':
        from n_body_system.se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        model = SE3_Transformer(n_particles=5, n_dimesnion=3, nf=int(args.nf/args.degree), n_layers=args.n_layers,
                                model=args.model, num_degrees=args.degree, div=1)
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        raise Exception("Wrong model specified")

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    train_loss_dict = {}
    for epoch in range(0, args.epochs):
        train_loss, _ = train(model, optimizer, epoch, loader_train)
        train_loss_dict[epoch] = train_loss
        if epoch % args.test_interval == 0:
            #train(epoch, loader_train, backprop=False)
            val_loss, _ = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, sample_losses = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))
            #mean_RTE = sum(sample_losses)/len(sample_losses)
            #sample_losses.sort()
            #median_RTE = sample_losses[int(len(sample_losses)/2)]
            #max_RTE = max(sample_losses)
            #delta_RTE = best_test_loss - best_val_loss
            #print("*** Mean RTE: %.5f \t Median RTE: %.5f \t Max RTE: %.5f \t Delta RTE: %.5f \t Best epoch %d" % (mean_RTE, median_RTE, max_RTE, delta_RTE, best_epoch))
        json_object = json.dumps(results, indent=4)
        with open(args.local_path + file_name + "/test_losses.json", "w") as outfile:
            outfile.write(json_object)
    with open(args.local_path + file_name + "/train_losses.json", "w") as trajfile:
            traj_obj = json.dumps(train_loss_dict)
            trajfile.write(traj_obj)
    #with open('./train_loss.pkl', 'wb') as file:
        #dump(train_loss_dict, file)
    return best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    sample_loss = []
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}
    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        if args.task == 'N_body_system':
            loc, vel, edge_attr, charges, loc_end, edge_mask = data
        elif args.task == 'MD22':
            loc, vel, edge_attr, charges, loc_end, vel_end, edge_mask = data

        # edges
        if args.task == 'N_body_system':
            edges_tensor = loader.dataset.get_edges(batch_size, n_nodes, edge_mask.squeeze(-1))
            edges = edges_tensor.tolist()
            for idx, item in enumerate(edges):
                edges[idx] = torch.tensor(item)
            edges = [edges[0].to(device), edges[1].to(device)]
            edge_mask_ = torch.gt(edge_mask, 0).to('cpu')
            edge_attr = edge_attr[edge_mask_].unsqueeze(-1)

        elif args.task == 'MD22':
            edges_tensor = loader.dataset.get_edges(batch_size, n_nodes, edge_mask.squeeze(-1))
            edges = edges_tensor.tolist()
            for idx, item in enumerate(edges):
                edges[idx] = torch.tensor(item)
            edges = [edges[0].to(device), edges[1].to(device)]
            edge_mask_ = torch.gt(edge_mask, 0).to('cpu')
            edge_attr = edge_attr[edge_mask_.squeeze(-1)]


        optimizer.zero_grad()

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
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
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
        elif args.model in ['clof', 'clof_vel', 'clof_vel_gbf']:
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr, n_nodes=n_nodes)
        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
        loss = loss_mse(loc_pred, loc_end)
        sample_loss.append(loss.item())
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        if batch_idx % args.log_interval == 0 and (args.model == "se3_transformer" or args.model == "tfn"):
            print('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))

    if backprop and epoch % 50 == 0:
        torch.save(model, args.local_path + file_name + '-' + str(args.epochs) + '.pt')

    if not backprop:
        prefix = "==>"
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], sample_loss


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
    print(args.gpu)
    print(torch.cuda.is_available())
    torch.cuda.set_device(args.gpu)
    if args.sweep_training:
        main_sweep()
    else:
        main()




