"""
Implementing the DCEM (Differential Cross Entropy Method) Sampling to predict
the optical flow which is further used to obtain velocity for the controller
"""

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Normal


class Model(nn.Module):
    def __init__(self, lstm_units: int = 4, seqlen: float = 7.5, n_sample: int = 16, n_elite: int = 4, top_k: int = 4):
        """
        Initialize the model.
        
        Args:
        - lstm_units (int): Number of LSTM units.
        - seqlen (float): Length of the sequence.
        - n_sample (int): Number of samples.
        - n_elite (int): Number of elite samples.
        - top_k (int): Number of top k samples.
        """
        super(Model, self).__init__()
        self.seqlen = seqlen
        self.lstm_units = lstm_units
        self.batch_size = 1
        self.veldim = 6

        init_mu = 0.5
        init_sigma = 0.5
        self.n_sample = n_sample
        self.n_elite = n_elite
        self.top_k = top_k  # New parameter for top-k velocities

        ### Initialise Mu and Sigma
        self.mu = init_mu * torch.ones((1, 6), requires_grad=True).cuda()
        self.sigma = init_sigma * torch.ones((1, 6), requires_grad=True).cuda()

        self.dist = Normal(self.mu, self.sigma)

        self.f_interm = []
        self.v_interm = []
        self.mean_interm = []
        
        # Neural network initialization
        self.init_neural_network_block()

        self.linear = nn.Linear(256, 6)
        self.pooling = torch.nn.AvgPool2d(kernel_size=1, stride=1)


    def init_neural_network_block(self) -> None:
        """
        Initialize the neural network block.
        """
        self.block = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )


    def process_flow_hat(self, Lsx: torch.Tensor, Lsy: torch.Tensor, vels: torch.Tensor, f12: torch.Tensor) -> torch.Tensor:
        """
        Process the optical flow.
        
        Args:
        - Lsx (torch.Tensor): Lsx tensor.
        - Lsy (torch.Tensor): Lsy tensor.
        - vels (torch.Tensor): Velocities tensor.
        - f12 (torch.Tensor): f12 tensor.
        
        Returns:
        - torch.Tensor: Processed optical flow tensor.
        """
        Lsx = Lsx.view(1, f12.shape[2], f12.shape[3], 6)
        Lsy = Lsy.view(1, f12.shape[2], f12.shape[3], 6)

        f_hat = torch.cat(
            (
                torch.sum(Lsx * vels, -1).unsqueeze(-1),
                torch.sum(Lsy * vels, -1).unsqueeze(-1),
            ),
            -1,
        )

        f_hat = self.pooling(f_hat.permute(0, 3, 1, 2))

        return f_hat
    
    def update_gaussian_parameters(self, vel: torch.Tensor) -> None:
        """
        Update Gaussian parameters according to the velocity 
        
        """
        mu_copy = self.mu.detach().clone()
        self.mu = vel  # New mean is the selected velocity 
        self.sigma = ((mu_copy - self.mu) ** 2).sqrt()  # New sigma is the difference between old mean and new mean

    def get_pred_flow(self, vel: torch.Tensor, Lsx: torch.Tensor, Lsy: torch.Tensor, horizon: float, f12: torch.Tensor) -> torch.Tensor:
        """
        Predicts optical flow and compares with the original flow and also
        compares and updates the gaussian distribution parameters accordingly. 
        
        Args:
        - vel (torch.Tensor): Velocity tensor.
        - Lsx (torch.Tensor): Lsx tensor.
        - Lsy (torch.Tensor): Lsy tensor.
        - horizon (float): Horizon value.
        - f12 (torch.Tensor): f12 tensor.
        
        Returns:
        - torch.Tensor: Predicted optical flow tensor.
        """
        # f12 -> target flow  

        vel = self.dist.rsample((self.n_sample,)).cuda()
        self.f_interm.append(self.sigma)
        self.mean_interm.append(self.mu)

        vel = self.block(vel.view(self.n_sample, 1, 6))  # pass through the neural network 
       
        vel = torch.sigmoid(self.linear(vel)).view(self.n_sample, 6)  # linear layer and sigmoid
        vel = vel.view(self.n_sample, 1, 1, 6) * 2 - 1 # normalising it so it stays in -1 to 1 


        ### Horizon Bit
        vels = vel * horizon if horizon >= 0 else vel * -horizon


        f_hat = self.process_flow_hat(Lsx, Lsy, vels, f12)

        if horizon < 0:

            loss_fn = torch.nn.MSELoss(reduction='none')
            loss = loss_fn(f_hat, f12) # MSE Loss 
            loss = torch.mean(loss.reshape(self.n_sample, -1), dim=1)
            sorted_loss, indices = torch.sort(loss)

            # Consider top-k velocities
            top_k_indices = indices[:self.top_k]
            top_k_velocities = vel[top_k_indices]

            # Average the top-k velocities
            mean_vel = torch.mean(top_k_velocities, dim=0)
            vel = mean_vel.view(6,)

            self.v_interm.append(vel)

        self.update_gaussian_parameters(vel)
        
        return f_hat


if __name__ == "__main__":
    vs = Model().to("cuda:0")
    ve = torch.zeros(6).to("cuda:0")
    print(list(vs.parameters()))
