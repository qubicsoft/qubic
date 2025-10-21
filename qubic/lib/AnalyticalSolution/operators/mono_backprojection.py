import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qubic.lib.AnalyticalSolution.graphs import healpix_graph
from qubic.lib.AnalyticalSolution.utils.losses import edge_mse_iqu  
from qubic.lib.AnalyticalSolution.utils.edges import build_coobs_edges
from qubic.lib.AnalyticalSolution.utils.maps import full_nest_to_ring


class TODToSky_PT(nn.Module):
    """
    Apply P^T then divide by coverage for I,Q,U. 
    Adjusts the scale constant of the first order correction in the Neumann series.
    """
    def __init__(self, P_operator, seen_indexes_ring):
        super().__init__()
        self.P_operator = P_operator
        ones_sky = np.ones(self.P_operator.shapeout)
        projection_cov = self.P_operator.T(ones_sky)[:,0] # use only coverage of I

        self.projection_cov = torch.tensor(projection_cov, dtype=torch.float32, requires_grad=False)
        #self.register_buffer("projection_cov", torch.as_tensor(projection_cov, dtype=torch.float32))

        self.seen_indexes_ring = seen_indexes_ring

    def forward(self, tod_tensor):
        B = tod_tensor.shape[0]
        outs = []
        for i in range(B):
            with torch.no_grad():
                sky_np = self.P_operator.T(tod_tensor[i].detach().cpu().numpy())
                sky = torch.from_numpy(sky_np).to(tod_tensor.device)
                sky[:, 0] /= self.projection_cov
                sky[:, 1] /= self.projection_cov
                sky[:, 2] /= self.projection_cov
                outs.append(sky)
        return torch.stack(outs, dim=0)  # (B, Npix, 3)


class ScaleBias(nn.Module):
    """
    Per-channel scale + bias (learnable rescale). 
    Used to find c_1 in c_1 P^T/P.1. 
    """
    def __init__(self, init_scale = torch.tensor(1e-3)):
        super().__init__()
        self.scale = nn.Parameter(torch.as_tensor(init_scale))
        self.bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.scale + self.bias


class TrainableInverseProjection_FirstOrderScale(nn.Module):
    """
    P^T / cov + simple learnable rescale per Stokes.
    Returns scaled skies on the observed subset (N_obs, 1) for I,Q,U. The correction is learned on Graphs.
    Returns a list of (I_s, Q_s, U_s) in nested ordering in observed pixels only.
    """
    def __init__(self, P_operator, seen_indexes_ring, nside = None):
        super().__init__()
        self.P_operator = P_operator
        if nside is None:
            nside = hp.npix2nside(P_operator.shapein[0])
        self.nside = nside

        self.register_buffer("seen_indexes_ring",
                             torch.as_tensor(seen_indexes_ring, dtype=torch.long))
        seen_nest = hp.ring2nest(nside, np.asarray(seen_indexes_ring))
        self.register_buffer("seen_indexes_nest",
                             torch.as_tensor(seen_nest, dtype=torch.long))

        n_pix = 12 * nside**2
        ring2nest = hp.ring2nest(nside, np.arange(n_pix))
        self.register_buffer("ring2nest",
                             torch.as_tensor(ring2nest, dtype=torch.long))

        self.reconstruction = TODToSky_PT(P_operator, seen_indexes_ring)
        self.lnn_I = ScaleBias(1e-3)
        self.lnn_Q = ScaleBias(1e-3)
        self.lnn_U = ScaleBias(1e-3)

    def forward(self, tod_tensor):
        outs = []
        sky_b = self.reconstruction(tod_tensor)       # (B, Npix, 3)
        device = sky_b.device

        for i in range(sky_b.shape[0]):
            # first go from RING to NEST full
            sky_i = sky_b[i]                          # (Npix,3) RING
            sky_n = torch.zeros_like(sky_i)
            sky_n[self.ring2nest] = sky_i        # (Npix,3) NEST
                       

            I = sky_n[self.seen_indexes_nest, 0]
            Q = sky_n[self.seen_indexes_nest, 1]
            U = sky_n[self.seen_indexes_nest, 2]

            #G = healpix_graph(nside=self.nside, nest=True)
            #Gp = G.subgraph(self.seen_indexes_nest.detach().cpu().numpy()) # Just observed pixels

            # Convert graph adjacency to edge_index for PyTorch Geometric
            #edge_index = torch.tensor(Gp.W.nonzero(), dtype=torch.long, device=device)

            # Scale to convenient units for training, nn works best when residuals are order around 1
            I_scaled = I * 1e18
            Q_scaled = Q * 1e21
            U_scaled = U * 1e21

            I_s = self.lnn_I(I_scaled.unsqueeze(-1))   # (N_obs,1)
            Q_s = self.lnn_Q(Q_scaled.unsqueeze(-1))
            U_s = self.lnn_U(U_scaled.unsqueeze(-1))

            # Back to original scale
            delta_I_hat = I_s / 1e18  
            delta_Q_hat = Q_s / 1e21
            delta_U_hat = U_s / 1e21

            outs.append((delta_I_hat, delta_Q_hat, delta_U_hat))
        return outs
    


def train_inverseprojection_firstorderscale(sky_list, tod_list,
                             P_operator, seen_indexes_ring, device, nside = None, 
                             n_epochs=100, optim_lr=1e-3, optim_weight_decay = 0, batch_size=1, print_grads = True):
    """
    Find the scale of first order correction with Adam.

    sky_list, tod_list : arrays
        Arrays of shape (num_samples, n_pix, 3) and (num_samples, n_det, n_time, 3).

    P_operator : callable
        Projection operator for one frequency.
    
    seen_indexes_ring : list or 1D array
        Pixel indices (in ring ordering) that are "observed".
    
    device : torch.device
        Device to run the training on ('cpu' or 'cuda').
    
    optim_lr, optim_weight_decay : float
        Learning parameters for Adam.
    
    batch_size : int
        Batch size. If num_samples < batch_size, sampling is done with replacement.

    Returns:
    --------
    model : TrainableInverseProjection_FirstOrderScale
        The trained inverse projection model.
    """


    num_samples = sky_list.shape[0]

    # Convert to torch Tensors
    sky_tensor = torch.tensor(np.array(sky_list), dtype=torch.float32, device=device) # (num_samples, n_pix, 3)
    tod_tensor = torch.tensor(np.array(tod_list), dtype=torch.float32, device=device)

    if nside is None:
        nside = hp.npix2nside(P_operator.shapein[0])
    
    # Initialize the trainable inverse projection model
    model = TrainableInverseProjection_FirstOrderScale(
        P_operator,
        seen_indexes_ring=seen_indexes_ring,
        nside=nside,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, weight_decay = optim_weight_decay)
    optimizer = torch.optim.Adam([
        {'params': model.lnn_I.scale, 'lr': 1e-1},
        {'params': model.lnn_I.bias, 'lr': 1e-3},
        {'params': model.lnn_Q.scale, 'lr': 1e-1},
        {'params': model.lnn_Q.bias, 'lr': 1e-3},
        {'params': model.lnn_U.scale, 'lr': 1e-1},
        {'params': model.lnn_U.bias, 'lr': 1e-3},
    ], lr=1e-3)

    # Convert seen indexes to torch on device (just for consistency in the training loop)
    seen_indexes_ring = torch.tensor(seen_indexes_ring, dtype=torch.long, device=device)
    n_pix = 12 * nside**2
    ring2nest_np = hp.ring2nest(nside, np.arange(n_pix))
    ring2nest_torch = torch.from_numpy(ring2nest_np).long().to(device)
    seen_indexes_nest = model.seen_indexes_nest.to(device)

    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample a batch from your dataset
        batch_indices = np.random.choice(num_samples, batch_size, replace=num_samples > batch_size)
        tod_batch = tod_tensor[batch_indices]  # (batch_size, n_det, n_time, 3)
        sky_batch = sky_tensor[batch_indices]  # (batch_size, n_pix, 3)

        # Forward pass: model returns a list of (x_I_hat, x_Q_hat, x_U_hat)
        reconstructed_skies = model(tod_batch)

        loss_total = 0.0

        # For each sample in the batch
        for i, (x_I_hat, x_Q_hat, x_U_hat) in enumerate(reconstructed_skies):
            # True sky for the same partial region (observed pixels):
            sky_ring_i = sky_batch[i]             # (n_pix, 3) in ring
            sky_nest_i = torch.zeros_like(sky_ring_i)
            sky_nest_i[ring2nest_torch] = sky_ring_i     # (n_pix, 3) in nest
            
            sky_true_I = sky_nest_i[seen_indexes_nest, 0]  # shape (num_seen,)
            sky_true_Q = sky_nest_i[seen_indexes_nest, 1]
            sky_true_U = sky_nest_i[seen_indexes_nest, 2]

            # GNN outputs are shape (num_seen, 1) - remove last dim
            pred_I = x_I_hat.squeeze(-1) 
            pred_Q = x_Q_hat.squeeze(-1) 
            pred_U = x_U_hat.squeeze(-1) 

            # The relative mse loss
            loss_I = torch.mean((pred_I - sky_true_I)**2) 
            loss_Q = torch.mean((pred_Q - sky_true_Q)**2) 
            loss_U = torch.mean((pred_U - sky_true_U)**2) 

            loss = loss_I + loss_Q + loss_U 
            loss_total += loss

        loss_total /= batch_size
        loss_total.backward()
        
        # Print gradients for all grad parameters
        #if print_grads:
            #for name, param in model.named_parameters():
            #   if param.grad is None:
            #      print(f"{name} has no grad!")
            # else:
                #    print(f"{name} grad mean = {param.grad.mean().item():.3e}")
        
        optimizer.step()
        if print_grads:
            print(f"Epoch {epoch}, Loss={loss_total.item():.6e}", " pred_I grad_fn =", pred_I.grad_fn)

    return model



class SharedCheb3Head(nn.Module):
    """
    Chebyshev body with 3 heads (I,Q,U). Residual output.
    """
    def __init__(self, hidden=128, K=3, L=4):
        super().__init__()
        self.enc0 = ChebConv(3, hidden, K)
        self.encs = nn.ModuleList([ChebConv(hidden, hidden, K) for _ in range(L-1)])
        self.headI = ChebConv(hidden, 1, K)
        self.headQ = ChebConv(hidden, 1, K)
        self.headU = ChebConv(hidden, 1, K)

    def forward(self, x, edge_index):
        h = F.relu(self.enc0(x, edge_index))
        for conv in self.encs:
            h = F.relu(conv(h, edge_index) + h)
        dI = self.headI(h, edge_index)
        dQ = self.headQ(h, edge_index)
        dU = self.headU(h, edge_index)
        return x + torch.cat([dI, dQ, dU], dim=1)


def build_graphs_IQU(scaled, orig, edge_index, tod_list, S_I = 1e17, S_QU = 1e18):
    """
    Create Data objects for training the graph network. 
    """
    out = []
    for i in range(len(scaled)):
        x = torch.stack([scaled[i,:,0]*S_I,
                         scaled[i,:,1]*S_QU,
                         scaled[i,:,2]*S_QU], dim=1)
        y = torch.stack([(orig[i,:,0]-scaled[i,:,0])*S_I,
                         (orig[i,:,1]-scaled[i,:,1])*S_QU,
                         (orig[i,:,2]-scaled[i,:,2])*S_QU], dim=1)
        tod_I = torch.tensor(tod_list[i,:,:,0], dtype=torch.float32).flatten()
        out.append(Data(x=x, y=y, edge_index=edge_index, tod=tod_I))
    return out


def train_cheb(net, loader, edge_geo, edge_coobs,
               S_I, S_QU,
               λ_pix=10., λ_geo=1., λ_coobs=1., λ_phys=0.,
               project_to_tod=None, npix=None, seen_nest=None, seen_ring=None,
               device='cpu', epochs=1000, lr=1e-3, step=1000, gamma=0.3, log_every=50):
    """
    Pixel MSE + geometric edge MSE + coobs edge MSE (+ optional physics loss).
    """
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=step, gamma=gamma)

    for epoch in range(epochs):
        net.train()
        tot = 0.0
        for batch in loader:
            batch = batch.to(device)
            optim.zero_grad()

            pred = net(batch.x, edge_geo)
            true = batch.x + batch.y

            loss_pix = 1000. * λ_pix   * F.mse_loss(pred, true)
            loss_geo = 1000. * λ_geo   * edge_mse_iqu(pred, true, edge_geo)
            loss_co  = 1000. * λ_coobs * edge_mse_iqu(pred, true, edge_coobs)

            loss = loss_pix + loss_geo + loss_co
            loss_phys = torch.tensor(0., device=device)

            if λ_phys > 0 and project_to_tod is not None:
                pred_phys = torch.stack([pred[:,0]/S_I, pred[:,1]/S_QU, pred[:,2]/S_QU], dim=1)
                full_ring = full_nest_to_ring(pred_phys, npix, seen_nest, seen_ring)
                tod_mod = torch.tensor(project_to_tod(full_ring.detach().cpu().numpy())[:,:,0].flatten(),
                                       dtype=torch.float32, device=device)
                loss_phys = F.mse_loss(tod_mod, batch.tod)
                loss += 1e30 * λ_phys * loss_phys

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optim.step()
            tot += loss.item()

        sched.step()
        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | total {tot/len(loader):.4e} | "
                  f"pix {loss_pix:.4e} | geo {loss_geo:.4e} | coobs {loss_co:.4e} | phys {loss_phys:.4e}")
