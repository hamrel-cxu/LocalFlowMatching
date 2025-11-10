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
from PIL import Image
from scipy.stats import chi2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### Data specific

def inf_train_gen(img_name, data_size):
    def gen_data_from_img(image_mask, train_data_size):
        ''' From FFJORD '''
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] # Pre-noise image
            samples = np.random.randn(*m.shape) * std + m # Add tiny noise
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        return full_data
    image_mask = np.array(Image.open(f'{img_name}.png').rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    return dataset

def get_x_raw(ntr, nte, dataname):
    datax = inf_train_gen(f'data/img_{dataname}', ntr+nte)
    data0 = torch.from_numpy(datax[:ntr]).float().to(device)
    xte = torch.from_numpy(datax[ntr:]).float().to(device)
    return data0, xte

def OU_at_t(x,t,z=None):
    if z is None:
        z = torch.randn_like(x).to(device)
    # OU process
    shrink = math.exp(-t)
    return shrink*x + math.sqrt(1-shrink**2)*z

#### Model specific
def build_MLP(in_features, hidden_features):
    layers = nn.ModuleList()
    i = 0
    for a, b in zip([in_features] + hidden_features, hidden_features + [in_features]):
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
        self.act = nn.Softplus(beta=20)
        self.int_mtd, self.rtol, self.atol = 'dopri5', 1e-5, 1e-5

    def forward(self, t, x):
        if len(t.size()) == 2:
            # During training, one t per x, so t is [batch_size, 1] with size = 2
            tt = t
        else:
            tt = torch.ones_like(x[:, :1]).to(device) * t
        for i, layer in enumerate(self.net):
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
        integration_time = torch.linspace(0.0, 1.0, num_steps).to(device)
        with torch.no_grad():
            z, ladj = tdeq.odeint(augmented, (x, ladj), integration_time, method=self.int_mtd, rtol = self.rtol, atol = self.atol)
        z = z[-1]
        ladj = ladj[-1]
        if return_full:
            # Final log-likelihood 
            log_qz = Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1)
            return log_qz + ladj
        else:
            # Just change by the current model
            return ladj

class InterFlowLoss(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x0, x1):
        self.v.train()
        pi = math.pi
        t = torch.rand_like(x0[:, :1]).to(device)
        if 'use_FM' in args_yaml:
            partial_It = x1 - x0
            It_x0x1 = x0 + t*partial_It
        else:
            cos_c, sin_c = torch.cos(pi*t/2).to(device), torch.sin(pi*t/2).to(device)
            It_x0x1 = cos_c*x0 + sin_c*x1
            partial_It = pi/2*(cos_c*x1 - sin_c*x0)
        vout = self.v(t, It_x0x1)
        loss_tk = (vout - partial_It).square().mean()
        return loss_tk

#### Visualization

def true_OU():
    # Plot true OU over all time steps
    _, xte = get_x_raw(ntr, nte, dataname)
    data_ls = [xte]
    for l in range(1, L+2):
        if l < L+1:
            data_ls.append(OU_at_t(data_ls[-1], hks[l-1]))
        else:
            data_ls.append(torch.randn_like(xte))
    data_ls = torch.stack(data_ls).cpu().numpy()
    num_cols = len(data_ls)
    fig, ax = plt.subplots(1, num_cols, figsize=(num_cols*4, 4), sharey=True, sharex=True)
    for i in range(num_cols):
        ax[i].scatter(data_ls[i, :, 0], data_ls[i, :, 1], s=s)
        if i == 0:
            title = 'Raw data'
        elif i < num_cols-1:
            title = f'OU at t={np.cumsum(hks[:i])[-1].item():.3f}'
        else:
            title = 'Standard normal'
        fsize = 26
        ax[i].set_title(title, fontsize=fsize)
    fig.savefig(os.path.join(master_dir, 'True_OU.png'), dpi=100)
    plt.close()

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

def viz_xy_hat(xte, yte, flow_ls, num_steps, batch = None):
    if batch is None:
        raise ValueError('Specify batch')    
    # NOTE, yte below is ALWAYS the Gaussian noise
    # xte may be pushed by previous blocks
    ####### Generating
    xte_raw = xte.cpu().numpy()
    # Forward, push raw x through previous blocks
    if l > 1:
        ypast = [xte.cpu().numpy()] # For visualizing trajectory
        ypast_blocks, xte = return_prev_block(xte, flow_ls, forward = True)
        ypast = ypast + ypast_blocks
        ypast = np.stack(ypast)
    # Current block (push x through current block or generate from standard Gaussian)
    with torch.no_grad():
        yhat_traj = flow.front_or_back(xte.to(device), towards_one = True).cpu().numpy()
        xhat_traj = flow.front_or_back(yte.to(device), towards_one = False).cpu().numpy()
    # Backward, generate through previous blocks
    if l > 1:
        xpast = return_prev_block(torch.from_numpy(xhat_traj[-1]).to(device), flow_ls, forward = False)[0]
        xpast = np.stack(xpast)
    ####### Collect results
    xhat = xhat_traj[-1] # Note, the xhat now is result from CURRENT latest block, not true xhat by all blocks
    yhat = yhat_traj[-1] # Pushed through all blocks so far
    # Results by each block
    if l > 1:
        yte_reshape = yte.reshape(1,*xte.shape).cpu().numpy()
        xhat_blocks = np.concatenate([yte_reshape, xhat.reshape(1,*xte.shape), xpast, xte_raw.reshape(1,*xte.shape)], 0)
        yhat_blocks = np.concatenate([ypast, yhat.reshape(1,*xte.shape), yte_reshape], 0)
    else:
        xhat_blocks = np.stack([xhat_traj[0], xhat, xte_raw])
        yhat_blocks = np.stack([yhat_traj[0], yhat, yte.cpu().numpy()])
    # Intermediate trajectory at current block (starting at its xte)
    if l < L+1:
        ytrue = OU_at_t(xte, hk).cpu().numpy()
    else:
        ytrue = torch.randn_like(xte).cpu().numpy()
    yhat_traj = np.concatenate([yhat_traj, ytrue.reshape(1,*xte.shape)], 0)
    ####### Start plotting
    grid_size = 4
    fsize = 26
    # Plot generated samples
    ncols = 3
    fig, ax = plt.subplots(1, ncols, figsize=(ncols * grid_size, 4))
    xhat = xpast[-1] if l > 1 else xhat
    ax[0].scatter(xhat[:, 0], xhat[:, 1], s=s)
    ax[0].set_title('Xhat', fontsize = fsize)
    ax[1].scatter(yhat[:, 0], yhat[:, 1], s=s)
    ax[1].set_title('Yhat', fontsize = fsize)
    ax[2].scatter(ytrue[:, 0], ytrue[:, 1], s=s)
    ax[2].set_title(f'Ytrue, Block {l}', fontsize = fsize)
    sub_dir = os.path.join(master_dir, f'Block_{l}', 'Generated')
    os.makedirs(sub_dir, exist_ok=True)
    fig.savefig(os.path.join(sub_dir, f'Gen_batch{batch}.png'), dpi=100)
    plt.close()
    # Plot results after each block
    fig, axs = plt.subplots(2, l+2, figsize=(grid_size*(l+2), grid_size*2), sharex = 'row', sharey = 'row')
    traj_dict = {0: yhat_blocks, 1: xhat_blocks}
    for i in range(2):
        traj = traj_dict[i]
        for j in range(l+2):
            ax = axs[i, j]
            ax.scatter(traj[j, :, 0], traj[j, :, 1], s=s)
            if j == 0:
                title = 'X' if i == 0 else 'Y'
            elif j == l+1:
                title = 'Y' if i == 0 else 'X'
            else:
                title = f'Block {j}' if i == 0 else f'Block {l-j+1}'
            ax.set_title(title, fontsize = fsize)
    fig.tight_layout()
    fig.savefig(os.path.join(sub_dir, f'Blocks_batch{batch}.png'), dpi=100)
    plt.close()
    # Plot trajectory (t=0 to t=1 only)
    fig, axs = plt.subplots(1, num_steps+1, figsize=(grid_size*(num_steps+1), grid_size*1), sharex = 'row', sharey = 'row')
    weights = np.linspace(0, 1, num_steps)
    traj = yhat_traj
    for j in range(num_steps+1):
        ax = axs[j]
        ax.scatter(traj[j, :, 0], traj[j, :, 1], s=s)
        if j == num_steps:
            title = 'Y'
            ax.set_title(title, fontsize = fsize)
        else:
            ax.set_title(f't={weights[j]:.2f}', fontsize = fsize)
    fig.tight_layout()
    fig.savefig(os.path.join(sub_dir, f'Traj_batch{batch}.png'), dpi=100)
    plt.close()

def viz_confidence_regions():
    def F(input):
        # Your existing transformation logic remains unchanged
        xhat = flow.front_or_back(input, towards_one = False)[-1]
        if l > 1:
            xhat = return_prev_block(xhat, flow_ls, forward = False)[0]
        return xhat[-1]
    alphas = np.linspace(0.05, 0.95, 6)  # Significance levels
    pts = inf_train_gen(f'img_{dataname}', 50000)  # True plots to be plotted
    angles = np.linspace(0, 2 * np.pi, 20000)  # Range of angles for the circles
    dim = 2  # Dimensionality of Z
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colormap = plt.cm.seismic  # Define the colormap
    for alpha in alphas:
        radius = np.sqrt(chi2.ppf(1 - alpha, df=dim))
        z_points = torch.tensor([radius * np.cos(angles), radius * np.sin(angles)]).T.float().to(device)
        p_points = F(z_points)
        # Map alpha to the corresponding color in the colormap
        color = colormap((1 - alpha - min(alphas)) / (max(alphas) - min(alphas)))
        # Plot with the color mapped from alpha
        plt.scatter(p_points[:, 0], p_points[:, 1], s=0.5, color=color)
    # Plot the true data points
    plt.scatter(pts[:,0], pts[:, 1], s=0.001, color='black')
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(alphas), vmax=max(alphas)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'Confidence Level $1-\alpha$', fontsize=24)
    cbar.set_ticks([alpha for alpha in alphas])
    cbar.set_ticklabels([f'{alpha:.2f}' for alpha in alphas])
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(master_dir, 'Conf_region.png'), bbox_inches='tight', pad_inches=0.02)
    plt.close()

#### Miscellaneous
def load_prev_blocks(block_id):
    flow_ls = []
    for l_now in range(1, block_id):
        flow = CNF(2, hidden_features=[hidden_dim] * num_hidden).to(device)
        sdict_path = os.path.join(master_dir, f'state_dict_block_{l_now}.pt')
        sdict = torch.load(sdict_path)
        flow.load_state_dict(sdict['state_dict'])
        flow_ls.append(flow)
    return flow_ls

def get_loglike():
    # Log-likelihood
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
    print(f'###### NLL after block {l} is {log_p:.3f}')
    return log_p

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
# Local Flow Matching
parser.add_argument('--hyper_param_config', default = 'config/2d_LFM_rose.yaml',
                    type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()
with open(args_parsed.hyper_param_config, 'r') as file:
    args_yaml = yaml.safe_load(file)
    print(yaml.dump(args_yaml, default_flow_style=False))

if __name__ == '__main__':
    # For saving and resuming
    dataname = args_yaml['data']['dataname']
    s = 0.00025
    master_dir = os.path.join('results_2d', args_yaml['save_dir'])
    os.makedirs(master_dir, exist_ok=True)
    # Data from p_X
    ntr, nte = args_yaml['data']['ntr'], args_yaml['data']['nte']
    #### Hyperparameters
    # Training
    batch_size = args_yaml['training']['batch_size']
    max_batch = args_yaml['training']['max_batch']
    resume = args_yaml['training']['resume']
    # Viz hyperparameters
    viz_freq = args_yaml['visualize']['viz_freq']
    num_steps = args_yaml['visualize']['num_steps']
    # Blocks 
    L = args_yaml['training']['L']
    hks = args_yaml['training']['hks']
    #### Visualize how hk is by checking TRUE OU
    true_OU()
    #### Training loop
    log_p_tot = []
    for l in range(1, L+2):
        flow_ls = None
        if l > 1:
            # Load previous blocks
            flow_ls = load_prev_blocks(block_id = l)
        hk = hks[l-1]
        print(f'####### Block {l} with hk = {hk}')
        # Data 
        if l == 1:
            # Get raw data for block 1
            data0, xte = get_x_raw(ntr, nte, dataname)
            yte = torch.randn_like(xte).to(device) # Always sample from standard Gaussian
        else:
            # Get input for next block
            with torch.no_grad():
                data0 = flow.front_or_back(data0.to(device), towards_one = True)[-1]        
        # Model for current block to be trained
        hidden_dim, num_hidden = args_yaml['model']['hidden_dim'], args_yaml['model']['num_hidden']
        flow = CNF(2, hidden_features=[hidden_dim] * num_hidden).to(device)
        if l > 1 and args_yaml['training']['warm_start']:
            flow.load_state_dict(flow_ls[-1].state_dict())
        print(flow)
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
        for batch in tqdm(range(start_batch, max_batch), ncols=88):
            subset = torch.randint(0, ntr, (batch_size,))
            x0 = data0[subset].to(device)
            z = torch.randn(batch_size, 2).to(device)
            xforx1 = x0
            if 'independent' in args_yaml['training']:
                subset_prime = torch.randint(0, ntr, (batch_size,))
                x0prime = data0[subset_prime].to(device)
                xforx1 = x0prime
            if l < L+1:
                x1 = OU_at_t(xforx1, hk, z) 
            else:
                if hk is not None:
                    # Not "free" block
                    x1 = OU_at_t(xforx1, hk, z)
                else:
                    x1 = z
            optimizer.zero_grad()
            loss_x = loss(x0, x1)
            loss_x.backward()
            loss_ls.append(loss_x.item())
            optimizer.step()
            scheduler.step()
            # Evaluation
            if batch % viz_freq == 0 or batch == max_batch - 1:
                # Print GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**2
                    reserved = torch.cuda.memory_reserved(device) / 1024**2
                    print(f"[GPU {device}] Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB (after batch {batch})")
                # Plot losses
                def loss_convolve(loss_ls, window_size):
                    if len(loss_ls) < window_size:
                        return loss_ls
                    else:
                        return scipy.signal.convolve(loss_ls, np.ones(window_size)/window_size, 
                                         mode='valid', method = 'fft')
                window_size = 500
                loss_ls_plot = loss_convolve(loss_ls, window_size = window_size)
                plt.plot(loss_ls_plot)
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                sub_dir = os.path.join(master_dir, f'Block_{l}', 'Losses')
                os.makedirs(sub_dir, exist_ok=True)
                plt.savefig(os.path.join(sub_dir, f'loss_batch{batch}.png'))
                plt.close()
                # Sampling (backward) & Forward push
                viz_xy_hat(xte, yte, flow_ls, num_steps, batch = batch)
                # Save model
                sdict = {'state_dict': flow.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'batch': batch+1,
                        'loss_ls': loss_ls}
                torch.save(sdict, sdict_path)
                plt.close('all')
                # Compute log-likelihood
                get_loglike()
        # Sampling (backward) & Forward push, after full training
        viz_xy_hat(xte, yte, flow_ls, num_steps, batch = '_final')
        # Compute log-likelihood
        log_p = get_loglike()
        log_p_tot.append(log_p)
        # Confidence region
        if l == L+1:
            viz_confidence_regions()
        plt.close('all')
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"[GPU {device}] Memory Allocated: {allocated:.2f} MB, Memory Reserved: {reserved:.2f} MB (after block {l})")
    print(f'NLL over all blocks is {np.round(log_p_tot, 3)}')