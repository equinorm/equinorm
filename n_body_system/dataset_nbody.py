import numpy as np
import torch
import random
import os
import pickle as pkl
from pytorch3d import transforms
from scipy.stats import ortho_group

def rot_or_ref(matrix):
    '''
    matrix: square matrix
    Determine whether the input matrix is rotation or reflection
    '''
    if torch.linalg.det(matrix) < 0:
        return 'reflection'
    else:
        return 'rotation'


class NBodyDataset():
    """
    NBodyDataset
    """
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer", cutoff=False, cutoff_lambda=None,
                 test_rot=False, test_trans=False, test_ref=False):
        self.cutoff = cutoff
        if self.cutoff:
            self.cutoff_lambda = cutoff_lambda

        # partition suffix
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        print('dataset name')
        print(dataset_name)
        self.dataset_name = dataset_name
        # dataset & task name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        elif dataset_name == "large20":
            self.sufix += "_charged20_initvel1large20"
        elif dataset_name == "large50":
            self.sufix += "_charged50_initvel1large50"
        elif dataset_name == "large100":
            self.sufix += "_charged100_initvel1large100"
        elif dataset_name == "large150":
            self.sufix += "_charged150_initvel1large150"
        elif dataset_name == "large200":
            self.sufix += "_charged200_initvel1large200"
        else:
            # raise Exception("Wrong dataset name %s" % self.dataset_name)
            self.sufix += dataset_name
        print('charge')
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.test_ref = test_ref

    def load(self):
        local_path = '/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/'
        loc = np.load(local_path + 'n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load(local_path + 'n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load(local_path + 'n_body_system/dataset/edges_' + self.sufix + '.npy')
        charges = np.load(local_path + 'n_body_system/dataset/charges_' + self.sufix + '.npy')

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):

        # pairwise distance
        #def d(_i, _j, _t):
            #print(loc.shape)
            #print(loc[0,0,0,:])
            #print(loc[0,:,0,0])
            #return np.sqrt(np.sum((np.array(loc[_t][_i]) - np.array(loc[_t][_j])) ** 2))
        #    return np.sqrt(np.sum((np.array(loc[:, _t, _i, :]) - np.array(loc[:, _t, _j, :])) ** 2, axis=1))
        # cast to torch and swap n_nodes <--> n_features dimensions
        def adj_to_list(adj):
            '''
            adj_tensor: [B, N, N]
            return: list [B*N*(N-1)]
            '''
            B, N, _ = adj.shape
            identity_mask = 1 - torch.eye(N) # [N, N]
            traj_mask = identity_mask.unsqueeze(0).repeat(B,1,1).reshape(B*N*N,) # [B*N*N]
            mask_list = torch.nonzero(traj_mask).squeeze(-1)
            adj_list = adj.reshape(B*N*N,) # [B*N*N]
            return adj_list[mask_list] # [B*edges]

        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        print(loc.shape)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []
        #Initialize edges and edge_attributes
        total_neighbors = 0
        rows, cols = [], []

        # fully connected
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j: # not self
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
                    total_neighbors += 1

        traj, _, _, _ = loc.shape
        self.edge_mask = torch.ones(traj, n_nodes * (n_nodes - 1))


        # cutoff
        if self.cutoff:
            frame_0 = loc[:, 0, :, :] # [B, N, 3]
            loc_i = frame_0.unsqueeze(2).repeat(1,1,n_nodes,1) # [B, N, N, 3]
            loc_j = frame_0.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # [B, N, N, 3]
            rel_loc = loc_i - loc_j # [B, N, N, 3]
            distance = np.linalg.norm(rel_loc, axis=-1) # [B, N, N]
            adj_tensor = torch.Tensor(np.where(distance < self.cutoff_lambda, 1, 0)) # [B, N, N]
            edge_mask =  adj_to_list(adj_tensor).squeeze(-1) # [B*num_edges]
            self.edge_mask = edge_mask.reshape(traj, -1) # [B, num_edges]
            #neigh_edge_index = torch.nonzero(edge_mask).squeeze(-1) # [B*neigh_edges]
            #self.neigh_edges = neigh_edge_index.reshape(traj, -1) # [B, neigh_edges]
            neigh_edge_index = torch.nonzero(edge_mask) # [B, neigh_edges]
            total_neighbors = neigh_edge_index.shape[0]/traj


        avg_neighbors = total_neighbors / n_nodes
        print('Average number of neighbors {:f}!'.format(avg_neighbors))
        # fully-connected edges
        edges = [rows, cols]
        edge_attr = np.array(edge_attr)
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension
        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    '''
    def preprocess_old(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc0 = loc[0:self.max_samples, 0, :, :]  # first location from the trajectory
        loc_last = loc[0:self.max_samples, -1, :, :]  # last location from the trajectory
        vel = vel[0:self.max_samples, 0, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc0), torch.Tensor(vel), torch.Tensor(edge_attr), loc_last, edges, torch.Tensor(charges)
    '''

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            # raise Exception("Wrong dataset partition %s" % self.dataset_name)
            frame_0, frame_T = 30, 40
        # input position, input velocity, edge attr, particle charges, output positions
        loc_0, vel_0, _, _, loc_t = loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]


        if self.test_rot and self.partition == 'test': # rotation
            rot = transforms.random_rotation() # random rotation
            assert rot_or_ref(rot) == 'rotation' # check rotation
            loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), rot.detach().numpy())) # rot input loc
            vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), rot.detach().numpy())) # rot input vel
            loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), rot.detach().numpy())) # rot output loc

        if self.test_trans and self.partition == 'test': # translation
            dimension = loc_t.max(dim=0)[0] - loc_t.min(dim=0)[0] #
            trans = torch.randn(3) * dimension / 2 # random translation
            loc_0 = loc_0 + trans # trans input loc
            loc_t = loc_t + trans # trans output loc

        if self.test_ref and self.partition == 'test': # reflection
            eye = torch.eye(3)
            eye[1, 1] = -1
            rot = transforms.random_rotation()  # random rotation
            ref = eye @ rot # random reflection
            assert rot_or_ref(ref) == 'reflection'  # check reflection
            loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), ref.detach().numpy()))  # ref input loc
            vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), ref.detach().numpy()))  # ref input vel
            loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), ref.detach().numpy()))  # ref output loc


        return loc_0, vel_0, edge_attr, charges, loc_t, self.edge_mask[i].unsqueeze(-1)

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes, edge_mask):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            # loop batch size
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
            edges_ = torch.stack(edges, dim=0)
            edge_mask_ = torch.gt(edge_mask, 0).to('cpu')
            edges_ = edges_[:,edge_mask_]
            return edges_


class MD22Dataset():
    """
    MD22 dataset, supra-molecules
    """
    def __init__(self, partition, max_samples, delta_frame, data_dir, molecule_type, cutoff=False, hydrogen_atoms=True,
                 test_rot=False, test_trans=False, test_ref=False, cutoff_lambda=1.6):
        # setup a split, tentative setting
        train_par, val_par, test_par = 0.8, 0.1, 0.1
        full_dir = os.path.join(data_dir, molecule_type + '.npz')
        split_dir = os.path.join(data_dir, molecule_type + '_split.pkl')
        data = np.load(full_dir)
        self.partition = partition
        self.molecule_type = molecule_type
        self.hydrogen_atoms = hydrogen_atoms

        #
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.test_ref = test_ref

        self.cutoff = cutoff
        if self.cutoff:
            self.cutoff_lambda = cutoff_lambda

        x = data['R']
        v = x[1:] - x[:-1] # velocity (t - t-1)
        x = x[:-1] # (0, t-1)
        try: # split file
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except: # random split
            np.random.seed(100) # random seed

            #_x = x[10000: -10000] # pick interval

            if x.shape[0] > 10000:
                trun_num = 3000
            else:
                trun_num = 100

            _x = x[trun_num: -trun_num]

            train_size = round(int(train_par * _x.shape[0]), -2)
            train_idx = np.random.choice(np.arange(_x.shape[0]), size=train_size, replace=False) # select train idx list
            flag = np.zeros(_x.shape[0]) # flag for train id
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]] # rest for val and test

            valid_size = round(int(val_par * _x.shape[0]), -2)
            val_idx = np.random.choice(rest, size=valid_size, replace=False) # select val id list
            for _ in val_idx: # flag for val
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]] # rest for test

            test_size = round(int(test_par * _x.shape[0]), -2)
            test_idx = np.random.choice(rest, size=test_size, replace=False) # select test id list

            # recover true indices and get split
            if x.shape[0] > 10000:
                train_idx += 3000
                val_idx += 3000
                test_idx += 3000
            else:
                train_idx += 100
                val_idx += 100
                test_idx += 100

            split = (train_idx, val_idx, test_idx)

            # save indices
            with open(split_dir, 'wb') as f:
                pkl.dump(split, f)
            print('Generate and save split!')

        # get id
        if partition == 'train':
            st = split[0]
        elif partition == 'val':
            st = split[1]
        elif partition == 'test':
            st = split[2]
        else:
            raise NotImplementedError()

        # filter max samples
        st = st[:max_samples]
        z = data['z']  # mol idx
        print('mol idx:', z)
        # whether filter hydrogen atoms
        if not self.hydrogen_atoms:
            x = x[:, z > 1, ...]
            v = v[:, z > 1, ...]
            z = z[z > 1]

        # position and velocity
        x_0, v_0 = x[st], v[st] # input
        x_t, v_t = x[st + delta_frame], v[st + delta_frame] # output

        print('Got {:d} samples!'.format(x_0.shape[0])) # number of samples
        print('Got {:d} particles!'.format(x_0.shape[1]))

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node = n_node

        _lambda = 3 # distance cutoff threshold

        # pairwise distance
        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]

        self.Z = torch.Tensor(z)

        # N * N adjacency matrix
        total_neighbors = 0
        atom_edges = torch.zeros(n, n).int()
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.cutoff: # local cutoff
                        _d = d(i, j, 0)
                        if _d < self.cutoff_lambda: # close pair
                            atom_edges[i][j] = 1
                            total_neighbors += 1
                    else: # fully connected
                        atom_edges[i][j] = 1
                        total_neighbors += 1
        #if self.cutoff: # cutoff, second order features
        #    atom_edges2 = atom_edges @ atom_edges
        #    self.atom_edge2 = atom_edges2
        self.atom_edge = atom_edges # [N, N]
        avg_neighbors = total_neighbors/n
        print('Average number of neighbors {:f}!'.format(avg_neighbors))

        self.edge_mask = torch.ones(n_node*(n_node - 1)) # [N, N-1] fully connected initial

        edge_attr = []
        rows, cols = [], []
        for i in range(n_node):
            for j in range(n_node):
                if i != j: # fully-connected
                    rows.append(i)
                    cols.append(j)
                    edge_attr.append([mole_idx[i], mole_idx[j], 1])
                    #if self.cutoff: # cutoff
                    #    if self.atom_edge[i][j]:
                    #        rows.append(i)
                    #        cols.append(j)
                    #        edge_attr.append([mole_idx[i], mole_idx[j], 1]) # bond type, first order
                            #assert not self.atom_edge2[i][j]  # check whether second order is zero
                    #    if self.atom_edge2[i][j]: # second order feature is not zero
                    #        rows.append(i)
                    #        cols.append(j)
                    #        edge_attr.append([mole_idx[i], mole_idx[j], 2])
                            #assert not self.atom_edge[i][j] # check whether no edges
                #else: # fully-connected
        if self.cutoff:
            N, _ = self.atom_edge.shape
            identity_mask = 1 - torch.eye(N)  # [N, N]
            traj_mask = identity_mask.reshape(N * N, ) # [N*N]
            mask_list = torch.nonzero(traj_mask).squeeze(-1) # [N]
            adj_list = self.atom_edge.reshape(N * N, )  # [N*N]
            edge_mask = adj_list[mask_list] # [num_edges]
            self.edge_mask = edge_mask

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        self.x_0, self.v_0, self.x_t, self.v_t = torch.Tensor(x_0), torch.Tensor(v_0), torch.Tensor(x_t), torch.Tensor(
            v_t) # positon, velocity attributes
        self.mole_idx = torch.Tensor(mole_idx) # atom types tensors


    def __getitem__(self, i):
        loc_0, vel_0, edge_attr, mole_idx, loc_t, vel_t =self.x_0[i], self.v_0[i], self.edge_attr, \
                                                         self.mole_idx.unsqueeze(-1), self.x_t[i], self.v_t[i]
        if self.test_rot and self.partition == 'test': # rotation
            print('rotating testing inputs')
            rot = transforms.random_rotation() # random rotation
            assert rot_or_ref(rot) == 'rotation' # check rotation
            loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), rot.detach().numpy())) # rot input loc
            vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), rot.detach().numpy())) # rot input vel
            loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), rot.detach().numpy())) # rot output loc
            vel_t = torch.tensor(np.matmul(vel_t.detach().numpy(), rot.detach().numpy())) # rot output vel

        if self.test_trans and self.partition == 'test': # translation
            print('translating testing inputs')
            dimension = loc_t.max(dim=0)[0] - loc_t.min(dim=0)[0] #
            trans = torch.randn(3) * dimension / 2 # random translation
            loc_0 += trans # trans input loc
            loc_t += trans # trans output loc

        if self.test_ref and self.partition == 'test': # reflection
            eye = torch.eye(3)
            eye[1, 1] = -1
            rot = transforms.random_rotation()  # random rotation
            ref = eye @ rot # random reflection
            print
            assert rot_or_ref(ref) == 'reflection'  # check reflection
            loc_0 = torch.tensor(np.matmul(loc_0.detach().numpy(), ref.detach().numpy()))  # ref input loc
            vel_0 = torch.tensor(np.matmul(vel_0.detach().numpy(), ref.detach().numpy()))  # ref input vel
            loc_t = torch.tensor(np.matmul(loc_t.detach().numpy(), ref.detach().numpy()))  # ref output loc
            vel_t = torch.tensor(np.matmul(vel_t.detach().numpy(), ref.detach().numpy()))  # ref output vel

        #if self.test_rot and self.partition == "test":
        #    d = np.random.randn(3)
        #    d = d / np.linalg.norm(d)
        #    angle = random.randint(0, 360)
        #    ts_0 = transformations.rotate.rotateby(angle, direction=d, ag=self.data.atoms)(ts_0)
        #if self.test_trans and self.partition == 'test':
        #    trans = np.random.randn(3) * ts_0.dimensions[:3] / 2
        #    ts_0 = transformations.translate(trans)(ts_0)

        #if self.test_rot and self.partition == "test":
        #    ts_t = transformations.rotate.rotateby(angle, direction=d, ag=self.data.atoms)(ts_t)
        #if self.test_trans and self.partition == 'test':
        #    ts_t = transformations.translate(trans)(ts_t)
        # initial positions, initial velocity, edge_attr, atom_types, output positions, output velocity
        return loc_0, vel_0, edge_attr, mole_idx, loc_t, vel_t, self.edge_mask.unsqueeze(-1)
               #self.Z.unsqueeze(-1)

    def __len__(self):
        # number of samples
        return len(self.x_0)

    def get_edges(self, batch_size, n_nodes, edge_mask):
        # get fully connected edges within random batch
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size): # batch of edges
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
            edges_ = torch.stack(edges, dim=0)
            edge_mask_ = torch.gt(edge_mask, 0).to('cpu')
            edges_ = edges_[:, edge_mask_]
        return edges_

if __name__ == "__main__":
    #NBodyDataset(dataset_name='large100', cutoff=True, cutoff_lambda=20)
    partition = 'train'
    max_samples = 5000
    delta_frame = 50
    data_dir = '/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/MD22/'
    #molecule_type = 'md22_AT-AT-CG-CG'
    #molecule_type = 'md22_buckyball-catcher'
    molecule_type = 'md22_dw_nanotube'
    cutoff = True
    cutoff_lambda=20
    dataset =MD22Dataset(partition=partition, max_samples=max_samples, delta_frame=delta_frame,
                         data_dir=data_dir, molecule_type=molecule_type, cutoff=cutoff, cutoff_lambda=cutoff_lambda)
    #print(len(dataset[7]))