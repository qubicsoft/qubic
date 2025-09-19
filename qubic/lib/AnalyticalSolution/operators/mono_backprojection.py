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


class TODToSky(nn.Module):
    """
    Apply P^T then divide by coverage for I,Q,U.
    """
    def __init__(self, P_operator, projection_cov, seen_indexes_ring):
        super().__init__()
        self.P_operator = P_operator
        self.register_buffer("projection_cov",
                             torch.as_tensor(projection_cov, dtype=torch.float32))
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
    """
    def __init__(self, init_scale):
        super().__init__()
        self.scale = nn.Parameter(torch.as_tensor(init_scale))
        self.bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index=None):
        return x * self.scale + self.bias


class TrainableInverseProjection(nn.Module):
    """
    P^T / cov + simple learnable rescale per Stokes.
    Returns scaled skies on the observed subset (N_obs, 1) for I,Q,U.
    """
    def __init__(self, P_operator, projection_cov, seen_indexes_ring, nside):
        super().__init__()
        self.P_operator = P_operator
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

        self.reconstruction = TODToSky(P_operator, projection_cov, seen_indexes_ring)
        self.lnn_I = ScaleBias(1e-3)
        self.lnn_Q = ScaleBias(1e-2)
        self.lnn_U = ScaleBias(1e-2)

    def forward(self, tod_tensor):
        outs = []
        sky_b = self.reconstruction(tod_tensor)       # (B, Npix, 3)
        device = sky_b.device

        for i in range(sky_b.shape[0]):
            sky_i = sky_b[i]                          # (Npix,3) RING
            sky_n = torch.zeros_like(sky_i)
            sky_n[self.ring2nest] = sky_i             # NEST full

            I = sky_n[self.seen_indexes_nest, 0]
            Q = sky_n[self.seen_indexes_nest, 1]
            U = sky_n[self.seen_indexes_nest, 2]

            G = healpix_graph(nside=self.nside, nest=True)
            Gp = G.subgraph(self.seen_indexes_nest.detach().cpu().numpy())
            edge_index = torch.tensor(Gp.W.nonzero(), dtype=torch.long, device=device)

            I_s = self.lnn_I(I.unsqueeze(-1), edge_index)   # (N_obs,1)
            Q_s = self.lnn_Q(Q.unsqueeze(-1), edge_index)
            U_s = self.lnn_U(U.unsqueeze(-1), edge_index)

            outs.append((I_s, Q_s, U_s))
        return outs


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


def build_graphs_IQU(scaled, orig, edge_index, tod_list, S_I, S_QU):
    """
    Pack graphs for PyG: x = scaled (in scaled units), y = residual (scaled).
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
