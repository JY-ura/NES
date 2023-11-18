from torch import nn
import torch
from typing import Tuple, List, Optional, Callable
from pathlib import Path

import torch.nn as nn
from utils.get_grad import ZOEstimator

USE_QUERY_RATIO = 0.5


class MetaZOEstimator(nn.Module):
    """
    A class representing a Meta Zeroth Order Optimization (Meta-ZO) estimator.

    Args:
    - sample_num (int): The number of samples to use for estimating the gradient.
    - max_sample_num_per_forward (int): The maximum number of samples to use per forward pass.
    - targeted (str): The type of targeted attack to perform.
    - grad_clip_threshold (float): The threshold for clipping the gradient.
    - device (torch.device): The device to use for computation.
    - flattened_input_dim (int): The flattened input dimension.
    - num_layers (int): The number of layers for the LSTM.
    - input_dim (int): The input dimension.
    - hidden_size (int): The hidden size of the LSTM.
    - normalize (bool): Whether to normalize the standard deviation.
    - checkpoint_path (Optional[str]): The path to the checkpoint.
    - reg_lambda (float): The regularization parameter.

    Methods:
    - load(checkpoint_path, freeze_update_rnn): Loads the model state from a checkpoint.
    - save(epoch, checkpoint_path, best): Saves the model state to a checkpoint.
    - reset_state(keep_states): Resets the state of the LSTM.
    - forward(grad): Performs a forward pass on the gradient.
    - zo_estimation(model, evaluate_img, target_labels, attack_loss_fun, skip_query_rnn, skip_update_rnn, lr): Performs a zeroth order optimization estimation.
    """

    def __init__(
        self,
        sample_num: int,
        max_sample_num_per_forward: int,
        targeted: str,
        grad_clip_threshold: float,
        device: torch.device,
        flattened_input_dim: int,
        attack_loss_fun: Callable,
        penalty_fun: Callable,
        penalty_eta: float = 0.1,
        num_layers: int = 1,
        input_dim: int = 1,
        hidden_size: int = 10,
        normalize: bool = True,
        reg_lambda: float = 0.1,
    ) -> None:
        """
        Initializes the Estimator class.

        Args:
        - sample_num (int): The number of samples to use for each forward pass.
        - max_sample_num_per_forward (int): The maximum number of samples to use for each forward pass.
        - targeted (str): The type of targeted attack to perform.
        - grad_clip_threshold (float): The threshold for gradient clipping.
        - device (torch.device): The device to use for computation.
        - flattened_input_dim (int): The flattened input dimension.
        - num_layers (int, optional): The number of layers for the LSTM. Defaults to 1.
        - input_dim (int, optional): The input dimension. Defaults to 1.
        - hidden_size (int, optional): The hidden size for the LSTM. Defaults to 10.
        - normalize (bool, optional): Whether to normalize the input. Defaults to True.
        - checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
        - reg_lambda (float, optional): The regularization parameter. Defaults to 0.1.
        """
        super().__init__()
        self.sample_num = sample_num
        self.max_sample_num_per_forward = max_sample_num_per_forward
        self.targeted = targeted
        self.grad_clip_threshold = grad_clip_threshold
        self.device = device
        self.normalize = normalize
        self.flattened_input_dim = flattened_input_dim
        self.reg_lambda = reg_lambda
        self.attack_loss_fun = attack_loss_fun
        self.penalty_eta = penalty_eta
        self.penalty_fun = penalty_fun

        # init state of the update rnn, query u rnn
        self.step = None
        self.query_u_rnn_state: Optional[Tuple[torch.Tensor,
                                               torch.Tensor]] = None
        self.update_rnn_state: Optional[Tuple[torch.Tensor,
                                              torch.Tensor]] = None

        self.update_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=False
        )
        self.update_projector = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=False
        )
        self.update_projector.weight.data.mul_(0.1)

        self.query_u_rnn = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=False
        )
        self.query_u_rnn_projector = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=False
        )
        self.query_u_rnn_projector.weight.data.mul_(0.1)

        # init state of the update rnn
        self.previous_grad_estimation = torch.zeros(
            size=(self.flattened_input_dim,), device=self.device, requires_grad=False)
        self.previous_param_update = torch.zeros(
            size=(self.flattened_input_dim,), device=self.device, requires_grad=False)

    def load(self, checkpoint_path, freeze_update_rnn: bool):
        """
        Loads a checkpoint from the given path and sets the model state dict accordingly.
        If `freeze_update_rnn` is True, freezes the parameters of the update RNN and update projector.

        Args:
        - checkpoint_path (str): Path to the checkpoint file.
        - freeze_update_rnn (bool): Whether to freeze the parameters of the update RNN and update projector.
        """
        assert Path(checkpoint_path).exists(
        ), f'Checkpoint path {checkpoint_path} does not exist.'
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if freeze_update_rnn:
            for param in self.update_rnn.parameters():
                param.requires_grad = False
            for param in self.update_projector.parameters():
                param.requires_grad = False
            print('Update RNN is frozen.')
        else:
            print('Update RNN is not frozen.')

    def save(self, epoch: int, checkpoint_path: str, best: bool):
        """
        Saves the model's state dictionary to a checkpoint file.

        Args:
            epoch (int): The current epoch number.
            checkpoint_path (str): The path to the directory where the checkpoint file will be saved.
            best (bool): If True, the checkpoint file will be named 'best.pt', otherwise it will be named '{epoch}.pt'.
        """
        assert Path(checkpoint_path).exists(
        ), f'Checkpoint path {checkpoint_path} does not exist.'
        if best:
            checkpoint_path = Path(checkpoint_path) / 'best.pt'
        else:
            checkpoint_path = Path(checkpoint_path) / f'{epoch}.pt'
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)

    def reset_state(self, keep_states: bool = False):
        """
        Resets the state of the estimator. If `keep_states` is False, the previous gradient estimation and parameter update
        are set to zero, and the query and update RNN states are initialized to zero. If `keep_states` is True, the query and
        update RNN states are detached from the computation graph, allowing them to be used for subsequent computations.

        Args:
            keep_states (bool): If True, the query and update RNN states are detached from the computation graph.
        """
        if not keep_states:
            self.previous_grad_estimation = torch.zeros(
                size=(self.flattened_input_dim,), device=self.device)
            self.previous_param_update = torch.zeros(
                size=(self.flattened_input_dim,), device=self.device)

            def init_lstm_state(lstm: nn.LSTM):
                hidden_size = lstm.hidden_size
                num_layers = lstm.num_layers
                batch_size = self.flattened_input_dim
                h = torch.zeros((num_layers, batch_size,
                                hidden_size), device=self.device)
                c = torch.zeros((num_layers, batch_size,
                                hidden_size), device=self.device)

                return h, c
            self.query_u_rnn_state = init_lstm_state(self.query_u_rnn)
            self.update_rnn_state = init_lstm_state(self.update_rnn)

        else:
            self.previous_grad_estimation = self.previous_grad_estimation.detach().clone()
            self.previous_param_update = self.previous_param_update.detach().clone()
            h, c = self.query_u_rnn_state
            self.query_u_rnn_state = (
                h.detach().clone(),
                c.detach().clone()
            )
            h, c = self.update_rnn_state
            self.update_rnn_state = (
                h.detach().clone(),
                c.detach().clone()
            )
            del h, c

        self.step = 0

    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the estimator.

        Args:
            grad (torch.Tensor): The gradient to be estimated.

        Returns:
            torch.Tensor: The estimated gradient after passing through the estimator.
        """
        grad = grad.reshape(-1, 1, 1)  # [batch_size, 1, 1]
        modified_grad, update_rnn_state = self.update_rnn(
            grad, self.update_rnn_state)
        self.update_rnn_state = update_rnn_state
        modified_grad = self.update_projector(modified_grad)
        return modified_grad.squeeze()

    def zo_estimation(
        self,
        model: nn.Module,
        evaluate_img: torch.Tensor,
        initial_img: torch.Tensor,
        target_labels: torch.Tensor,
        skip_query_rnn: bool = False,
        skip_update_rnn: bool = False,
        lr: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the zeroth-order gradient estimation for the given model and input image.

        Args:
            model (nn.Module): The model to estimate the gradient for.
            evaluate_img (torch.Tensor): The input image to estimate the gradient for.
            target_labels (torch.Tensor): The target labels for the attack.
            skip_query_rnn (bool, optional): Whether to skip the query RNN step. Defaults to False.
            skip_update_rnn (bool, optional): Whether to skip the update RNN step. Defaults to False.
            lr (float, optional): The learning rate to use for the update step when skip update rnn. Defaults to 0.001.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the estimated gradient,
            the true loss of the attack, and the regularization loss.
        """
        self.step += 1
        # use previous grad estimation and param update to estimate the grad
        x = torch.cat((self.previous_grad_estimation.unsqueeze(
            1), self.previous_param_update.unsqueeze(1)), dim=1)
        x = x.unsqueeze(1)
        x, query_u_rnn_state = self.query_u_rnn(
            x, self.query_u_rnn_state)
        self.query_u_rnn_state = query_u_rnn_state
        sigma = self.query_u_rnn_projector(x)
        sigma = sigma.squeeze()

        self.std = sigma
        self.mean = torch.zeros_like(self.std)
        regularize_loss = self.reg_lambda * (
            torch.sum(self.std ** 2) + torch.sum(self.mean ** 2))
        self.std = self.std + 1.0

        if self.normalize:
            self.std = self.std / self.std.norm() * torch.ones_like(self.std).norm()

        if self.training:
            use_query = True
        else:
            use_query = torch.rand(1) < USE_QUERY_RATIO

        grad_accumulation_steps = self.sample_num // self.max_sample_num_per_forward
        grad_list = []

        def _attack_loss(evaluate_imgs):
            evaluate_imgs = evaluate_imgs.reshape(-1, *evaluate_img.shape[1:])
            prediction = model(evaluate_imgs)
            return self.attack_loss_fun(prediction, target_labels.repeat(evaluate_imgs.size(0), 1), self.targeted)

        for _ in range(grad_accumulation_steps):
            perturbation = torch.randn(
                (self.max_sample_num_per_forward // 2, self.flattened_input_dim),
                device=self.device
            )
            if not skip_query_rnn and use_query:
                perturbation = self.std * perturbation + self.mean

            grad, loss = _symmetric_differentiation(
                _attack_loss, evaluate_img, perturbation)
            grad_list.append(grad)

        total_grad = torch.mean(torch.stack(grad_list), dim=0)
        penalty = self.penalty_fun(evaluate_img, initial_img, self.penalty_eta)

        true_loss = _attack_loss(evaluate_img) + penalty
        del _attack_loss

        if not skip_update_rnn:
            delta = self(total_grad)
        else:
            delta = - total_grad * lr

        self.previous_grad_estimation = total_grad.detach().clone()
        self.previous_param_update = delta.detach().clone()

        return delta, true_loss, regularize_loss


def _symmetric_differentiation(
    loss_fn: Callable,
    input: torch.Tensor,
    perturbation: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the symmetric finite difference of the loss function.

    Args:
        loss_fn (Callable): Loss function to be evaluated.
        input (torch.Tensor): Input image to be attacked.
        perturbation (torch.Tensor): Perturbation to be added to the input with shape [sample_num, *input.shape]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Gradient from finite difference of the loss function and the loss.
    """
    perturbation = torch.cat([perturbation, -perturbation], dim=0)
    adv_image = torch.clip(input.flatten() + perturbation, -0.5, 0.5)
    losses = loss_fn(adv_image)
    # losses [2 * sample_num]
    # compute hardmard product of losses and perturbation
    losses = losses.view(-1, 1,)
    grad = torch.mean(losses * perturbation, dim=0)
    loss = torch.mean(losses)
    return grad, loss
