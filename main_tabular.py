import math
import matplotlib.pyplot as plt
import argparse
import scipy
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from tqdm import tqdm
from typing import *
import torchdiffeq as tdeq
import os
import data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1103); np.random.seed(1103)
#### Data specific

def get_x_raw(dataname):
    if dataname == 'miniboone':
        data0, xte = data.tensor_high_dim('miniboone')
    elif dataname == 'bsds300':
        data0, xte = data.tensor_high_dim('bsds300')
    elif dataname == 'power':
        data0, xte = data.tensor_high_dim('power')
    elif dataname == 'gas':
        data0, xte = data.tensor_high_dim('gas')
    else:
        raise NotImplementedError
    print(f'True shape of train: {data0.shape}, test shape: {xte.shape}')
    return data0.to(device), xte.to(device)

def OU_at_t(x,t):
    if diff_batch:
        # See InterFlow (I.1)
        z = torch.randn(target_batch_size, x.shape[1]).to(device)
        x = x.repeat_interleave(target_batch_size, dim=0)
        z = z.repeat(base_batch_size, 1)
    else:
        z = torch.randn_like(x).to(device)
    if l < L+1:
        # OU process
        shrink = math.exp(-t)
        return x, shrink*x + math.sqrt(1-shrink**2)*z
    else:
        return x, z

#### Model specific
def build_MLP(in_features, hidden_features):
    layers = nn.ModuleList()
    i = 0
    for a, b in zip([in_features] + hidden_features, hidden_features + [in_features]):
        if 'cat_t' in args_yaml['model']:
            # +1 is for time in every layer
            a += 1
        else:
            if i == 0:
                # +1 is for time at the beginning
                a += 1
        layers.append(nn.Linear(a, b))
        i += 1
    return layers
    

class CNF(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.net = build_MLP(in_features, hidden_features)
        self.act = nn.ReLU() if dataname != 'bsds300' else nn.ELU()
        self.int_mtd, self.rtol, self.atol = 'dopri5', 1e-5, 1e-5

    def forward(self, t, x):
        if len(t.size()) == 2:
            # During training, one t per x, so t is [batch_size, 1] with size = 2
            tt = t
        else:
            tt = torch.ones_like(x[:, :1]).to(device) * t
        for i, layer in enumerate(self.net):
            if 'cat_t' in args_yaml['model']:
                x = torch.cat([tt, x], 1)   
            else:
                if i == 0:
                    x = torch.cat([tt, x], 1)            
            x = layer(x)
            if i < len(self.net) - 1:
                x = self.act(x)
        return x

    def front_or_back(self, z, towards_one = False):
        self.net.eval()
        first, second = 1, 0
        if towards_one:
            first, second = 0, 1
        integration_time = torch.linspace(first, second, num_steps).to(device)
        with torch.no_grad():
            return tdeq.odeint(self, z, integration_time, method=self.int_mtd, rtol = self.rtol, atol = self.atol)

    def log_prob(self, x, return_full = False):
        self.net.eval()
        I = torch.eye(x.shape[-1]).to(device)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)
        def augmented(t, x_ladj):
            x, ladj = x_ladj
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)
            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)
            return dx, trace
        ladj = torch.zeros_like(x[..., 0]).to(device)
        integration_time = torch.linspace(0.0, 1.0, 150).to(device)
        with torch.no_grad():
            z, ladj = tdeq.odeint(augmented, (x, ladj), integration_time, method='euler', rtol = self.rtol, atol = self.atol)
        z = z[-1]
        ladj = ladj[-1]
        if return_full:
            # Final log-likelihood 
            log_qz = Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1)
            # print(f'log_qz: {log_qz.mean().item()}, ladj: {ladj.mean().item()}')
            # print(f'min log_qz: {log_qz.min().item()}')
            return log_qz + ladj
        else:
            # Just change by the current model
            return ladj

class InterFlowLoss(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x0, x1, num_tk):
        self.v.train()
        loss_tk = 0
        pi = math.pi
        for _ in range(num_tk):
            if 'beta_sample_t' in args_yaml['training']:
                alpha, beta = args_yaml['training']['alpha'], args_yaml['training']['beta']
                beta_dist = torch.distributions.beta.Beta(alpha, beta)
                t = beta_dist.sample((x0.shape[0], 1)).to(device)
            else:
                t = torch.rand_like(x0[:, :1]).to(device)
            if 'use_FM' in args_yaml:
                sigma_min = args_yaml['use_FM']['sigma_min']
                It_x0x1 = (1-(1-sigma_min)*t)*x0 + t*x1
                partial_It =  x1 - (1-sigma_min)*x0
            else:
                sin_c, cos_c = torch.sin(pi*t/2).to(device), torch.cos(pi*t/2).to(device)
                It_x0x1 = cos_c*x0 + sin_c*x1
                partial_It = -pi/2*(x0*sin_c - x1*cos_c)
            vout = self.v(t, It_x0x1)
            loss_tk += (vout - partial_It).square().mean()
        return loss_tk/num_tk

class InterFlowLoss_new(nn.Module):
    # See InterFlow (I.1)
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x0, x1, num_tk):
        self.v.train()
        loss_tk = 0
        pi = math.pi
        for _ in range(num_tk):
            if 'beta_sample_t' in args_yaml['training']:
                alpha, beta = args_yaml['training']['alpha'], args_yaml['training']['beta']
                beta_dist = torch.distributions.beta.Beta(alpha, beta)
                t = beta_dist.sample().to(device)
            else:
                t = torch.rand(1).to(device)
            if 'use_FM' in args_yaml:
                sigma_min = args_yaml['use_FM']['sigma_min']
                It_x0x1 = (1-(1-sigma_min)*t)*x0 + t*x1
                partial_It =  x1 - (1-sigma_min)*x0
            else:
                sin_c, cos_c = torch.sin(pi*t/2).to(device), torch.cos(pi*t/2).to(device)
                It_x0x1 = cos_c*x0 + sin_c*x1
                partial_It = -pi/2*(x0*sin_c - x1*cos_c)
            vout = self.v(t, It_x0x1)
            loss_tk += (vout - partial_It).square().sum()/len(x0)
        return loss_tk/num_tk


#### Visualization
def return_prev_block(input, flow_ls, forward = True):
    xfull = []
    if forward:
        for flow in flow_ls:
            with torch.no_grad():
                input = flow.front_or_back(input.to(device), towards_one = True)[-1]
                xfull.append(input.cpu().numpy())
    else:
        for flow in flow_ls[::-1]:
            with torch.no_grad():
                input = flow.front_or_back(input.to(device), towards_one = False)[-1]
                xfull.append(input.cpu().numpy())
    return xfull, input

def plot_loss(loss_ls, batch):
    def loss_convolve(loss_ls, window_size = 500):
        if len(loss_ls) < window_size:
            return loss_ls
        else:
            return scipy.signal.convolve(loss_ls, np.ones(window_size)/window_size, 
                                mode='valid', method = 'fft')
    loss_ls_plot = loss_convolve(loss_ls)
    plt.plot(loss_ls_plot)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    sub_dir = os.path.join(master_dir, f'Block_{l}', 'Losses')
    os.makedirs(sub_dir, exist_ok=True)
    plt.savefig(os.path.join(sub_dir, f'loss_batch{batch}.png'))
    plt.close()

#### Miscellaneous
def load_prev_blocks(block_id):
    flow_ls = []
    for l_now in range(1, block_id):
        flow = CNF(data_dim, hidden_features=[hidden_dim] * num_hidden).to(device)
        sdict_path = os.path.join(master_dir, f'state_dict_block_{l_now}.pt')
        sdict = torch.load(sdict_path)
        flow.load_state_dict(sdict['state_dict'])
        flow_ls.append(flow)
    return flow_ls

def count_params(model):
    tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot_params/1000

def get_NLL(xte, flow_ls = None):
    # Log-likelihood
    def by_batches(xte):
        with torch.no_grad():
            xte_now = xte.clone().to(device)
            log_p_prev = 0
            if l > 1:
                for flow_prev in flow_ls:
                    div_f = flow_prev.log_prob(xte_now.to(device)).mean().item()
                    log_p_prev += div_f
                    xte_now = flow_prev.front_or_back(xte_now.to(device), towards_one = True)[-1]
            log_p = flow.log_prob(xte_now.to(device), return_full = True).mean().item()
            log_p += log_p_prev # Add previous divergences
            log_p = -log_p # NLL
        return log_p
    log_p = []
    max_bsize = 5000
    for i in range(0, len(xte), max_bsize):
        log_p.append(by_batches(xte[i:i+max_bsize]))
    log_p = np.mean(log_p).item()
    print(f'###### NLL at block {l} is {log_p:.3f}')
    return log_p

def push_by_current(flow, input):
    with torch.no_grad():
        max_size = 50000
        if len(input) <= max_size:
            # Push all at once
            return flow.front_or_back(input.to(device), towards_one = True)[-1]   
        else:
            # Push by chunks due to memory
            input_ls = []
            bsize_dict = {6: 60000, 8: 60000, 63: 30000}
            max_size = bsize_dict[data_dim]
            num_b = np.arange(0, len(input), max_size)
            for i in num_b:
                print(f'Pushing {i+1}/{len(input)}')
                input_ls.append(flow.front_or_back(input[i:i+max_size].to(device), towards_one = True)[-1])
            return torch.cat(input_ls, 0)

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--hyper_param_config', default = 'config/tabular_LFM_gas.yaml',
                    type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()
with open(args_parsed.hyper_param_config, 'r') as file:
    args_yaml = yaml.safe_load(file)
    print(yaml.dump(args_yaml, default_flow_style=False))

if __name__ == '__main__':
    # For saving and resuming
    dataname = args_yaml['data']['dataname']
    data_dim = args_yaml['data']['data_dim']
    master_dir = os.path.join('results_tabular', args_yaml['save_dir'])
    os.makedirs(master_dir, exist_ok=True)
    #### Hyperparameters
    # Training
    diff_batch = 'diff_batch' in args_yaml['training']
    if diff_batch:
        base_batch_size = args_yaml['training']['base_batch_size']
        target_batch_size = args_yaml['training']['target_batch_size']
    else:
        batch_size = args_yaml['training']['batch_size']
    max_batch_ls = args_yaml['training']['max_batch']
    resume = args_yaml['training']['resume']
    # Viz hyperparameters
    viz_freq = args_yaml['visualize']['viz_freq']
    num_steps = args_yaml['visualize']['num_steps']
    # Blocks 
    L = args_yaml['training']['L']
    hks = args_yaml['training']['hks']
    #### Training loop
    log_p_tot_tr = []
    log_p_tot = []
    for l in range(1, L+2):
        ### Load prev blocks
        flow_ls = None
        if isinstance(max_batch_ls, int):
            max_batch = max_batch_ls
        else:
            max_batch = max_batch_ls[l-1]
        if l > 1:
            flow_ls = load_prev_blocks(block_id = l)
        hk = hks[l-1]
        print(f'####### Block {l} with hk = {hk}')
        #### Data 
        if l == 1:
            # Get raw data for block 1
            data0, xte = get_x_raw(dataname)
            xtr = data0.clone().to(device)
            ntr = len(data0)
        else:
            # Get input for next block
            data0 = push_by_current(flow, data0)      
        ### Set up current model
        hidden_dim, num_hidden = args_yaml['model']['hidden_dim'], args_yaml['model']['num_hidden']
        flow = CNF(data_dim, hidden_features=[hidden_dim] * num_hidden).to(device)
        print(f'Number of parameters: {count_params(flow):.3f}k')
        if l > 1 and args_yaml['training']['warm_start']:
            flow.load_state_dict(flow_ls[-1].state_dict())
        print(flow)
        if diff_batch:
            loss = InterFlowLoss_new(flow).to(device) # Loss function depending on the flow model
        else:
            loss = InterFlowLoss(flow).to(device) # Loss function depending on the flow model
        optimizer = torch.optim.Adam(flow.parameters(), lr=args_yaml['training']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=args_yaml['training']['lr_step'], 
                                                    gamma=args_yaml['training']['lr_decay'])
        start_batch = 0
        loss_ls = []
        # Resume training
        sdict_path = os.path.join(master_dir, f'state_dict_block_{l}.pt')
        if os.path.exists(sdict_path) and resume:
            sdict = torch.load(sdict_path)
            flow.load_state_dict(sdict['state_dict'])
            optimizer.load_state_dict(sdict['optimizer'])
            scheduler.load_state_dict(sdict['scheduler'])
            start_batch = sdict['batch']
            loss_ls = sdict['loss_ls']   
        # Training current block
        for batch in tqdm(range(start_batch, max_batch), ncols=88):
            bsize = base_batch_size if diff_batch else batch_size
            subset = torch.randint(0, ntr, (bsize,))
            x0 = data0[subset].to(device)
            # Draw new noise for each batch
            x0, x1 = OU_at_t(x0, hk)
            optimizer.zero_grad()
            loss_x = loss(x0, x1, num_tk = args_yaml['training']['num_tk'])
            loss_x.backward()
            loss_ls.append(loss_x.item())
            optimizer.step()
            scheduler.step()
            # Evaluation
            if (batch % viz_freq == 0 or batch == max_batch-1) and batch > 0:
                # Print GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**2
                    reserved = torch.cuda.memory_reserved(device) / 1024**2
                    print(f"[GPU {device}] Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB (after batch {batch})")
                # Save model
                sdict = {'state_dict': flow.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'batch': batch+1,
                        'loss_ls': loss_ls}
                torch.save(sdict, sdict_path)
                # Plot losses                
                plot_loss(loss_ls, batch = batch)
                if batch < max_batch-1:
                    # Evaluate NLL
                    print('####### Training NLL #######')
                    _ = get_NLL(xtr[torch.randperm(len(xtr))[:min(20000, len(xte))]], flow_ls = flow_ls)
                    print('####### Test NLL #######')
                    _ = get_NLL(xte[torch.randperm(len(xte))[:min(20000, len(xte))]], flow_ls = flow_ls)
        ### After full training
        # Plot losses
        plot_loss(loss_ls, batch = '_final')
        # Evaluate NLL
        print('####### Training NLL #######')
        log_p_tr = get_NLL(xtr[torch.randperm(len(xtr))[:len(xte)]], flow_ls = flow_ls)
        log_p_tot_tr.append(log_p_tr)
        print('####### Test NLL #######')
        log_p = get_NLL(xte, flow_ls = flow_ls)
        log_p_tot.append(log_p)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"[GPU {device}] Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB (after block {l})")
    print(f'Training NLL over all blocks is {np.round(log_p_tot_tr, 3)}')
    print(f'Test NLL over all blocks is {np.round(log_p_tot, 3)}')