import torch
from typing import Tuple
import torch.nn as nn
from optimizer import margin_loss
import math

class ZOEstimator(nn.Module):
    def __init__(
        self,
        model,
        sample_num: int,
        max_sample_num_per_forward: int,
        sigma: float,
        targeted: str,
        grad_clip_threshold: float,
        device: torch.device,
        alpha: float=0.5,
        subspace_dim: int=10,
    ) -> None:
        super().__init__()
        assert targeted in ['targeted', 'untargeted']

        self.model = model
        self.sigma = sigma
        self.targeted = targeted
        self.grad_clip_threshold = grad_clip_threshold
        self.subspace_dim = subspace_dim
        self.device = device
        self.loss_func = margin_loss
        
        self.sample_num = sample_num
        self.alpha = alpha
        self.max_sample_num_per_forward = max_sample_num_per_forward
        self.previous_grad_queue = []
        self.previous_grads = None

    def forward(self, evaluate_img, target_labels):
        return self(evaluate_img, target_labels)

    def zo_estimation(
        self,
        evaluate_img: torch.Tensor,
        target_labels: torch.Tensor,
        subspace_estimation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_forwards = self.sample_num // self.max_sample_num_per_forward
        total_grads = []
        total_loss = []

        for _ in range(num_forwards):
            # determine wether to use subspace to sample perturbation
            if not subspace_estimation or len(self.previous_grad_queue) < self.subspace_dim:
                perturbation = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(self.max_sample_num_per_forward // 2,) +
                    evaluate_img.shape[1:],
                    device=self.device
                ) * self.sigma

            else:
                # print("subspace estimation")
                # self.previous_grad_queue.append(self.previous_grads.flatten())
                # if len(self.previous_grad_queue) == self.subspace_dim:

                previous_grads = torch.stack(
                    self.previous_grad_queue, dim=0).to(self.device)
                perturbation = self.subspace_sample(
                    previous_grads.transpose(0, 1),
                    evaluate_img.flatten().shape[0],
                    self.sigma
                )
                perturbation = perturbation.view(
                    (perturbation.size(0),) + evaluate_img[0].size())

            # generate + delta and - delta for evaluation
            perturbation = torch.concat([perturbation, -perturbation], dim=0)

            evaluate_imgs = evaluate_img + perturbation

            prediction = self.model(evaluate_imgs)
            loss = self.loss_func(prediction, target_labels, self.targeted)

            loss = loss.reshape(-1, 1, 1, 1).repeat(evaluate_img.shape)
            grad = loss * perturbation / self.sigma / 2

            total_grads.append(torch.mean(grad, dim=0, keepdim=True))
            total_loss.append(torch.mean(loss))
            # torch.cuda.empty_cache()

        if len(self.previous_grad_queue) == self.subspace_dim:
            self.previous_grad_queue.pop(0)

        total_grads = torch.mean(torch.concat(
            total_grads), dim=0, keepdim=True)
        if torch.norm(total_grads, p=2) > self.grad_clip_threshold:
            total_grads = total_grads / \
                torch.norm(total_grads, p=2)*self.grad_clip_threshold
        total_loss = torch.mean(torch.tensor(
            total_loss, device=self.device), dim=0)

        # self.previous_grads = total_grads.detach().clone()
        return total_grads, total_loss.detach()

    def subspace_sample(self, grads: torch.Tensor, d: int, sigma: float) -> torch.Tensor:
        """sample perturbation from subspace

        Args:
            grads (Tensor): previous gradients, used to formulate subspace
            d (int): dimension of the optimization space
            sigma (float): standard deviation of the gaussian noise

        Returns:
            torch.Tensor: perturbation sampled from subspace
        """
        perturbation_full_space = torch.randn(
            self.max_sample_num_per_forward//2, d, 1, device=self.device)
        perturbation_subspace = torch.randn(
            self.max_sample_num_per_forward//2, self.subspace_dim, 1, device=self.device)

        # use qr decomposition to get orthonormal basis
        q, _ = torch.qr(grads)
        perturbation = math.sqrt(sigma) * torch.sqrt(torch.tensor((1-self.alpha) / d)) * perturbation_full_space \
            + math.sqrt(sigma) * torch.sqrt(torch.tensor(self.alpha/self.subspace_dim)) * torch.matmul(q, perturbation_subspace)
        # q: [n, subspace_dim]
        # perturbation_subspace: [max_sample_num, subspace_dim, 1]
        return perturbation.squeeze(-1)

    def compute_true_loss(self, evaluate_img: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        score = self.model(evaluate_img)
        loss = self.loss_func(score, target_labels, self.targeted)
        return torch.squeeze(loss, 0)
