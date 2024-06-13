import torch
import numpy as np
import random


def process_md22(ratio):
    '''
    processing md22 trajectory
    param: ratio
    '''
    local_path = '/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/MD22/md22_'
    supra_mol_list = ['AT-AT-CG-CG', 'AT-AT', 'Ac-Ala3-NHMe','DHA', 'buckyball-catcher', 'dw_nanotube', 'stachyose']
    for supra_mol in supra_mol_list:
        tmp_path = local_path + supra_mol + '.npz'
        dataset = np.load(tmp_path)
        traj = dataset['R']  # [steps, atoms, 3]
        print(traj.shape)
        steps, _, _ = traj.shape
        if steps > 10000: # time interval
            T = 500
        else:
            T = 100
        # get train, valid, test indices
        id_list = [i for i in range(steps-T)]
        random.shuffle(id_list)
        id_arr = np.asarray(id_list)
        train_id_start = id_arr[0:ratio[0]]
        valid_id_start = id_arr[ratio[0]:ratio[0]+ratio[1]]
        test_id_start = id_arr[ratio[0]+ratio[1]:ratio[1]+ratio[2]]
        train_id_end = train_id_start + T
        valid_id_end = valid_id_start + T
        test_id_end = test_id_start + T
        # retrieve conformers with id
        train_input = traj[train_id_start, :, :]
        train_output = traj[train_id_end, :, :]
        valid_input = traj[valid_id_start, :, :]
        valid_output = traj[valid_id_end, :, :]
        test_input = traj[test_id_start, :, :]
        test_output = traj[test_id_end, :, :]
        # save dataset file



def process_chignolin(ratio):
    '''
    processing chignolin trajectory
    '''
    return

def process_md17(ratio):
    '''
    processing MD17 trajectory
    '''
    local_path = '/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/MD22/md17_'
    mol_list = ['aspirin', 'benzene2017', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']
    for mol in mol_list:
        tmp_path = local_path + mol + '.npz'
        dataset = np.load(tmp_path)
        traj = dataset['R']  # [steps, atoms, 3]
        #print(traj.shape)
        steps, _, _ = traj.shape

        x = dataset['R']  # pos
        v = x[1:] - x[:-1]  # vel
        x = x[:-1]  # (no last pos)
        z = dataset['z']  # mol idx
        print('mol idx:', z)

        # filter hydrogen atoms
        x = x[:, z > 1, ...]
        v = v[:, z > 1, ...]
        z = z[z > 1]

if __name__ == "__main__":
    ratio = [500, 2000, 2000] # train, val, test
    process_md22(ratio)
    #process_chignolin(ratio)
    #process_md17(ratio)


