import torch
from torch import nn
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel, Clof_GCL
import numpy as np
import logging

eps = 1e-5

def orthogonal_check(X):
    '''
    matrix: batched square matrix X, [b,3,3]
    Determine whether the input matrix is orthogonal
    '''
    return torch.bmm(X, X.transpose(1,2)) # [b,3,3]


def unique_direction(x_pca, U):
    '''
    x_pca: [B, N, 3]
    U: [B, 3, 3]
    return U
    '''
    b, l, _ = x_pca.shape
    std_x = torch.norm((x_pca).reshape(b * l, 3), dim=1, keepdim=True).squeeze(-1)  # [B*N]
    std_x = std_x.reshape(b, l)  # [B, N]
    _, max_indices = torch.max(std_x, dim=1, keepdim=True)  # [B,1]
    d = torch.gather(x_pca, 1, max_indices.unsqueeze(-1).repeat(1,1,3)) # [B,1,3]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    angle = cos(U.transpose(1,2).reshape(b*3, 3), d.repeat(1,3,1).reshape(b*3, 3)) # [B*3]
    angle = torch.where(angle>0, 1.0, -1.0) # [B*3]
    U = U * (angle.reshape(b,3).unsqueeze(1).repeat(1,3,1)) # [B, 3, 3]
    return U

def GSP(nearest_neigh_diff, eps=1e-5):
    '''
    Gram-schmidt process
    param: nearest_neigh_diff [B, 3, 3]
    return uu [B, 3, 3]: global coordinate system
    '''
    def projection(u, v):
        # project v on u: (v^{T}u/(u^T)u) * u
        return ((v * u).sum(-1).unsqueeze(-1).repeat(1, 3) / ((u * u).sum(-1).unsqueeze(-1).repeat(1, 3)+eps)) * u

    _, nk, _ = nearest_neigh_diff.shape
    uu = torch.zeros_like(nearest_neigh_diff, device=nearest_neigh_diff.device) # zero matrix [B, 3, 3]
    uu[:, 0, :] = nearest_neigh_diff[:, 0, :].clone() # clone first neighbor for each atom
    for k in range(1, nk): # loop other neighbors
        vk = nearest_neigh_diff[:, k, :].clone() # vk [B, 3]
        uk = 0 # init uk
        for j in range(0, k): # loop previous k vectors
            uj = uu[:, j, :].clone() # uj
            uk = uk + projection(uj, vk) # project vk on uj
        uu[:, k, :] = vk - uk # orthogonal
    # orthonormalization for direction
    for k in range(nk): # loop nk
        uk = uu[:, k, :].clone() # uk
        uu[:, k, :] = uk / (uk.norm(dim=-1).unsqueeze(-1).repeat(1, 1, 3)+eps) # norm uk
    return uu

class Norm(nn.Module):
    def __init__(self, norm_type, hidden_dim, n_body, batch_size, sigma=None, local_frame=None, device='cpu'):
        '''
        LN: vanilla LN
        VFS_norm: vector feature scaling norm, separate learnable g
        Get_norm: generalist norm, shared learnable g
        Equi_norm: shared learnable g and b
        '''
        super(Norm, self).__init__()
        # assert norm_type in ['layer_norm', 'scale_norm', 'equi_norm']
        self.norm_type = norm_type
        self.device = device
        self.n_body = n_body
        self.batch_size = batch_size
        l = int(n_body/batch_size)
        if self.norm_type == 'layer_norm': # LN
            self.norm_layer = nn.LayerNorm(normalized_shape=3, elementwise_affine=True)
        elif self.norm_type == 'VFS_norm': # VFS_g
            self.g = nn.Parameter(torch.randn(n_body, )) # B*N*1
        #elif self.norm_type == 'equi_norm_scalar':
        #    self.g = nn.Parameter(torch.randn(n_body, )) # B*N * 1
        #    self.b = nn.Parameter(torch.randn(n_body, )) # B*N * 1
        #elif self.norm_type == 'geo_norm': # geo_norm
        #    if sigma == 'norm':
        #        self.g = nn.Parameter(torch.randn(n_body, ))  # B*N * 1
        #    else:
        #        self.g = nn.Parameter(torch.randn(batch_size, ).repeat(l)) # B*N * 1
        #    self.b = nn.Parameter(torch.randn(n_body, 3))  # B*N * 1
        #    self.sigma = sigma
        #    self.local_frame = local_frame
        elif self.norm_type == "get_norm": # generalist_norm:
            self.g = nn.Parameter(torch.randn(self.batch_size, ))  # B*1
        elif self.norm_type == "equi_norm": # equi_norm
            self.sigma = sigma
            self.local_frame = local_frame
            self.g = nn.Parameter(torch.randn(self.batch_size, ))  # B*1
            self.b = nn.Parameter(torch.randn(self.batch_size, 3))  # B*3
            if self.local_frame == 'GSP':
                self.proj = nn.Parameter(torch.randn(n_body, 3)) # B*N,3
        self.to(self.device)
    
    def forward(self, x, x0=None):
        '''
        normalizaton layers
        x: [B*N, 3]
        '''
        if self.norm_type == 'layer_norm':
            return self.norm_layer(x)
        elif self.norm_type == 'VFS_norm':
            return self.g.unsqueeze(1) * (x/torch.norm(x, dim=1, keepdim=True))
        elif self.norm_type == 'get_norm':
            x = x.reshape(self.batch_size, -1, 3)  # [B, N, 3]
            b, l, _ = x.shape
            mu_x = torch.mean(x, dim=1, keepdim=True)  # [B, 1, 3]
            mu_x = mu_x.repeat(1, l, 1)  # [B, N, 3]
            std_x_in = (torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True)**2).squeeze(-1) # [B*N]
            std_x = torch.sqrt(std_x_in.reshape(b, l).sum(dim=1).reshape(b,)) # [B*1]
            x = x.reshape(b * l, 3)  # [B*N, 3]
            mu_x = mu_x.reshape(b * l, 3)  # [B*N, 3]
            return self.g.unsqueeze(1).repeat(l,1) * ((x - mu_x)/(std_x.unsqueeze(1).repeat(l,1)+1e-5)) + mu_x
        elif self.norm_type == 'equi_norm':
            x = x.reshape(self.batch_size, -1, 3)  # [B, N, 3]
            b, l, _ = x.shape
            mu_x = torch.mean(x, dim=1, keepdim=True) # [B, 1, 3]
            mu_x = mu_x.repeat(1, l, 1) # [B, N, 3]
            # scaling factor

            if self.sigma == 'std':
                std_x_in = (torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True) ** 2).squeeze(-1)  # [B*N]
                std_x_in = std_x_in.reshape(b, l).softmax(dim=1) + eps # [B, N]
                std_x = torch.sqrt(std_x_in.sum(dim=1).reshape(b, ) / l)  # [B*1]
                #std_x = torch.sqrt(std_x_in.reshape(b, l).sum(dim=1).reshape(b, )/l)  # [B*1]
            elif self.sigma == 'min-max':
                std_x = torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True).squeeze(-1) # [B*N]
                std_x = std_x.reshape(b,l) # [B, N]
                std_x = std_x.softmax(dim=1) + eps # [B, N]
                max_x, _ = torch.max(std_x, dim=1, keepdim=True) # [B,1]
                min_x, _= torch.min(std_x, dim=1, keepdim=True) # [B,1]
                std_x = (max_x - min_x).reshape(b,) # [B*1]

            # local frame
            if self.local_frame == 'PCA':
                x_pca = x - mu_x # [B, N, 3]
                cov_m = torch.bmm(x_pca.transpose(1,2), x_pca) # [B, 3, 3]
                #eigenvalues, eigenvectors = torch.linalg.eig(cov_m) # [B, 3], [B, 3, 3]
                cov_m = cov_m.detach().cpu().numpy()
                cov_m = np.where(np.isnan(cov_m), 0, cov_m)
                cov_m = np.where(np.isinf(cov_m), 0, cov_m)
                eigenvalues, eigenvectors = np.linalg.eig(cov_m)  # [B, 3], [B, 3, 3]
                eigenvectors = torch.from_numpy(eigenvectors).to(self.device)
                #eigenvalues, eigenvectors = torch.eig(cov_m)  # [B, 3], [B, 3, 3]
                frame = unique_direction(x_pca, eigenvectors) # [B, 3, 3]
                #eigenvectors = eigenvectors.unsqueeze(1).repeat(1,l,1,1) # [B, l, 3, 3]
                #frame = eigenvectors.reshape(b,3,3) # [B, 3, 3]
            elif self.local_frame == 'GSP':
                x_gsp = x - mu_x # [B, N, 3]
                init_frame = torch.bmm(self.proj.reshape(b,l,3).transpose(1,2), x_gsp) # [B, 3, 3]
                #id_list = np.arange(l)
                #np.random.shuffle(id_list)
                #frame_0 = frame_list[:, id_list[0], :].unsqueeze(1) # [B, 1, 3]
                #frame_1 = frame_list[:, id_list[1], :].unsqueeze(1) # [B, 1, 3]
                #frame_2 = frame_list[:, id_list[2], :].unsqueeze(1) # [B, 1, 3]
                #init_frame = torch.cat((frame_0, frame_1, frame_2), dim=1) # [B, 3, 3]
                init_frame = init_frame/init_frame.norm(dim=2).unsqueeze(-1).repeat(1,1,3)
                frame = GSP(init_frame).transpose(1,2) # [B, 3, 3]
                #frame = frame.unsqueeze(1).repeat(1,l,1,1) # [B, l, 3, 3]
                #frame = frame.reshape(b*l, 3, 3) # [B*l, 3, 3]

            # norm layer
            x = x.reshape(b*l, 3)  # [B*N, 3]
            mu_x = mu_x.reshape(b*l, 3)  # [B*l, 3]
            bias_proj = torch.bmm(frame, self.b.reshape(b,3).unsqueeze(-1)) # [B, 3, 1]
            bias_proj = bias_proj.squeeze(-1) # [B,3]
            #print(self.g.unsqueeze(1).repeat(l, 1) * ((x - mu_x) /std_x.unsqueeze(1).repeat(l, 1)).shape)
            #print(bias_proj.repeat(l,1).shape)
            #print(mu_0.shape)
            if x0 is not None:
                x0 = x0.reshape(self.batch_size, -1, 3)  # [B, N, 3]
                mu_0 = torch.mean(x0, dim=1, keepdim=True)  # [B, 1, 3]
                mu_0 = mu_0.repeat(1, l, 1)  # [B, N, 3]
                mu_0 = mu_0.reshape(b * l, 3)  # [B*l, 3]
                return self.g.unsqueeze(1).repeat(l, 1) * ((x - mu_x) /std_x.unsqueeze(1).repeat(l, 1)) + bias_proj.repeat(l,1) + mu_0 # [B*N,3]
            else:
                return self.g.unsqueeze(1).repeat(l, 1) * ((x - mu_x) /std_x.unsqueeze(1).repeat(l, 1)) + bias_proj.repeat(l,1)  # [B*N,3]

        #elif self.norm_type == 'equi_norm_scalar':
        #    x0 = x0.reshape(self.batch_size, -1, 3) # [B, N, 3]
        #    x = x.reshape(self.batch_size, -1, 3) # [B, N, 3]
        #    b, l, _ = x.shape
        #    mu_0 = torch.mean(x0, dim=1, keepdim=True) # [B, 1, 3]
        #    mu_x = torch.mean(x, dim=1, keepdim=True) # [B, 1, 3]
        #    mu_0 = mu_0.repeat(1, l, 1)  # [B, N, 3]
        #    mu_x = mu_x.repeat(1, l, 1)  # [B, N, 3]
        #    std_x_1 = torch.norm((x-mu_x).reshape(b*l, 3), dim=1, keepdim=True).repeat(1,3) # [B*N, 3]
        #    std_x_2 = torch.norm((mu_x-mu_0).reshape(b*l, 3), dim=1, keepdim=True).repeat(1,3) # [B*N, 3]
        #    mu_x = mu_x.reshape(b*l, 3) # [B*N, 3]
        #    mu_0 = mu_0.reshape(b*l, 3) # [B*N, 3]
        #    x = x.reshape(b*l, 3) # [B*N, 3]
        #    return self.g.unsqueeze(1) * ((x - mu_x) / std_x_1) + ((mu_x - mu_0) / std_x_2) * self.b.unsqueeze(1) + mu_0
        #elif self.norm_type == 'geo_norm':
        #    x0 = x0.reshape(self.batch_size, -1, 3)  # [B, N, 3]
        #    x = x.reshape(self.batch_size, -1, 3)  # [B, N, 3]
        #    b, l, _ = x.shape
        #    mu_0 = torch.mean(x0, dim=1, keepdim=True)  # [B, 1, 3]
        #    mu_x = torch.mean(x, dim=1, keepdim=True) # [B, 1, 3]
        #    mu_0 = mu_0.repeat(1, l, 1)  # [B, N, 3]
        #    mu_x = mu_x.repeat(1, l, 1) # [B, N, 3]
        #    if self.sigma == 'norm':
        #        std_x = torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True).repeat(1,3)  # [B*N, 3]
        #    elif self.sigma == 'std':
        #        std_x = torch.square(torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True)) # [B*N, 1]
        #        std_x = torch.sqrt(std_x.reshape(b,l,1).sum(dim=1)) # [B, 1, 1]
        #        std_x = std_x.repeat(1, l, 3).reshape(b*l, 3) # [B*N, 3]
        #    elif self.sigma == 'min-max':
        #        std_x = torch.norm((x - mu_x).reshape(b * l, 3), dim=1, keepdim=True).repeat(1,3) # [B*N, 3]
        #        std_x = std_x.reshape(b,l,3) # [B,N,3]
        #        max_x, _ = torch.max(std_x, dim=1, keepdim=True) # [B,1,3]
        #        min_x, _= torch.min(std_x, dim=1, keepdim=True) # [B,1,3]
        #        std_x = (max_x - min_x).repeat(1,l,1).reshape(b*l,3) # [B*N, 3]


            # get local frame
        #    if self.local_frame == 'PCA':
        #        x_pca = x - mu_x # [B, N, 3]
        #        cov_m = 1 / l * torch.bmm(x_pca.transpose(1,2), x) # [B, 3, 3]
                #eigenvalues, eigenvectors = torch.linalg.eig(cov_m) # [B, 3], [B, 3, 3]
        #        cov_m = cov_m.detach().cpu().numpy()
        #        eigenvalues, eigenvectors = np.linalg.eig(cov_m)  # [B, 3], [B, 3, 3]
        #        eigenvectors = torch.from_numpy(eigenvectors).to(self.device)
                #eigenvalues, eigenvectors = torch.eig(cov_m)  # [B, 3], [B, 3, 3]
        #        eigenvectors = eigenvectors.unsqueeze(1).repeat(1,l,1,1) # [B, l, 3, 3]
        #        frame = eigenvectors.reshape(b*l,3,3) # [B*l, 3, 3]
        #    elif self.local_frame == 'GSP':
        #        frame_list = x - mu_x # [B, N, 3]
        #        id_list = np.arange(l)
        #        np.random.shuffle(id_list)
        #        frame_0 = frame_list[:, id_list[0], :].unsqueeze(1) # [B, 1, 3]
        #        frame_1 = frame_list[:, id_list[1], :].unsqueeze(1) # [B, 1, 3]
        #        frame_2 = frame_list[:, id_list[2], :].unsqueeze(1) # [B, 1, 3]
        #        frame_set = torch.cat((frame_0, frame_1, frame_2), dim=1) # [B, 3, 3]
        #        frame = GSP(frame_set) # [B, 3, 3]
        #        frame = frame.unsqueeze(1).repeat(1,l,1,1) # [B, l, 3, 3]
        #        frame = frame.reshape(b*l, 3, 3) # [B*l, 3, 3]
        #    elif self.local_frame == 'CP':
        #        frame_list = x - mu_x  # [B, N, 3]
        #        id_list = np.arange(l)
        #        np.random.shuffle(id_list)
        #        frame_0 = frame_list[:, id_list[0], :].unsqueeze(1)  # [B, 1, 3]
        #        frame_1 = frame_list[:, id_list[1], :].unsqueeze(1)  # [B, 1, 3]
        #        partial_frame_set = torch.cat((frame_0, frame_1), dim=1)  # [B, 2, 3]
        #        partial_frame = GSP(partial_frame_set)  # [B, 2, 3]
        #        frame_2 = np.cross(partial_frame[:, 0, :].detach().cpu().numpy(),
        #                                     partial_frame[:, 1, :].detach().cpu().numpy()) # [B, 3]
        #        frame_2 = torch.from_numpy(frame_2).to(self.device)
        #        frame_2 = frame_2.unsqueeze(1) # [B, 1, 3]
        #        full_frame_set = torch.cat((partial_frame, frame_2), dim=1)  # [B, 3, 3]
        #        frame = full_frame_set.unsqueeze(1).repeat(1, l, 1, 1)  # [B, l, 3, 3]
        #        frame = frame.reshape(b * l, 3, 3)  # [B*l, 3, 3]
            # norm layer
        #    mu_0 = mu_0.reshape(b*l, 3)  # [B*N,3]
        #    bias_proj = torch.bmm(frame, self.b.unsqueeze(-1)) # [B*l,3,1]
        #    bias_proj = bias_proj.squeeze(-1) # [B*l,3]
        #    return self.g.unsqueeze(1) * ((x - mu_x).reshape(b*l,3) / std_x) + bias_proj + mu_0


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        #self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)


    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        #h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        #return h
        return self.decoder(h)


def get_velocity_attr(loc, vel, rows, cols):
    #return  torch.cat([vel[rows], vel[cols]], dim=1)

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va

class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=1.0):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        self.to(self.device)


    def forward(self, h, x, edges, edge_attr, vel=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            #if vel is not None:
                #vel_attr = get_velocity_attr(x, vel, edges[0], edges[1])
                #edge_attr = torch.cat([edge_attr0, vel_attr], dim=1).detach()
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        return x


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 batch_size=100, recurrent=False, norm_diff=False, tanh=False, norm_type=None, n_body=None, sigma='std', local_frame='PCA'):
        # in_node_nf 1, in_edge_nf 2, hidden_nf 64
        super(EGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.norm_type = norm_type
        self.n_body = n_body
        self.batch_size = batch_size
        if self.norm_type == 'equi_norm':
            self.sigma = sigma
            self.local_frame = local_frame
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh))
            if self.norm_type != 'None': # determine norm types
                if self.norm_type == 'equi_norm':
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, self.sigma, self.local_frame, device=self.device))
                else:
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, device=self.device))
        self.to(self.device)


    def forward(self, h, x, edges, vel, edge_attr):
        h = self.embedding(h)
        #x = x.reshape(-1, self.n_body, 3) # [B, N, 3]
        x0 = x.clone()
        # EGNN block
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
            if self.norm_type != 'None':
                #if self.norm_type == 'equi_norm_scalar':
                #    x = self._modules["norm_%d" % i](x, x0)
                if self.norm_type == 'equi_norm':
                    x = self._modules["norm_%d" % i](x, x0)
                else: # layer_norm, VFS_norm, get_norm
                    x = self._modules["norm_%d" % i](x)
        return x


class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), batch_size=100, n_layers=4,
                 norm_type=None, n_body=None, sigma='std', local_frame='PCA'):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.norm_type = norm_type
        self.n_body = n_body
        self.batch_size = batch_size
        if self.norm_type == 'equi_norm':
            self.sigma = sigma
            self.local_frame = local_frame
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
            if self.norm_type != 'None': # determine norm types
                if self.norm_type == 'equi_norm':
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, self.sigma, self.local_frame, device=self.device))
                else:
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, device=self.device))
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        x0 = x.clone()
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
            if self.norm_type != 'None':
                #if self.norm_type == 'equi_norm_scalar':
                #    x = self._modules["norm_%d" % i](x, x0)
                if self.norm_type == 'equi_norm':
                    x = self._modules["norm_%d" % i](x, x0)
                else: # layer_norm, VFS_norm, get_norm
                    x = self._modules["norm_%d" % i](x)
        return x


class ClofNet_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), batch_size=100, n_layers=4,
                 coords_weight=1.0, recurrent=True, norm_diff=True, tanh=False, norm_type=None, n_body=None,
                 sigma='std', local_frame='PCA'):
        super(ClofNet_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)

        edge_embed_dim = 16
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = True

        self.norm_type = norm_type
        self.n_body = n_body
        self.batch_size = batch_size

        if self.norm_type == 'equi_norm':
            self.sigma = sigma
            self.local_frame = local_frame

        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
            if self.norm_type != 'None': # determine norm types
                if self.norm_type == 'equi_norm':
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, self.sigma, self.local_frame, device=self.device))
                else:
                    self.add_module("norm_%d" % i, Norm(self.norm_type, self.hidden_nf, self.n_body, self.batch_size, device=self.device))
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges, x, vel):
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        r_i = x[row]
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)
        vel_i = torch.matmul(edge_basis,
                             v_i.unsqueeze(-1)).squeeze(-1)
        vel_j = torch.matmul(edge_basis,
                             v_j.unsqueeze(-1)).squeeze(-1)
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j],
                              dim=-1)  #[E, 14]
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)
            if self.norm_type != 'None':
                #if self.norm_type == 'equi_norm_scalar':
                #    x = self._modules["norm_%d" % i](x, x0)
                if self.norm_type == 'equi_norm':
                    x = self._modules["norm_%d" % i](x_center)
                else: # layer_norm, VFS_norm, get_norm
                    x = self._modules["norm_%d" % i](x_center)
        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x

class Baseline(nn.Module):
    def __init__(self, device='cpu'):
        super(Baseline, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc

class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device='cpu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)

class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1)*0.7)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v*self.time