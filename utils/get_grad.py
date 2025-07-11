import torch
from typing import Tuple
import torch.nn as nn
from optimizer import margin_loss, l2_regular_loss
from typing import Optional
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
        attack_loss: callable=margin_loss,
        penalty_fun: callable=l2_regular_loss,
        penalty_eta: float=0.1,
    ) -> None:
        """
        Zeroth Order Estimator for Black-box Adversarial Attacks.

        Args:
            model (nn.Module): PyTorch model to be attacked.
            sample_num (int): Number of samples to estimate the gradient.
            max_sample_num_per_forward (int): Maximum number of samples to be evaluated in one forward pass.
            sigma (float): Standard deviation of the Gaussian noise added to the input.
            targeted (str): Whether the attack is targeted or untargeted.
            grad_clip_threshold (float): Threshold for gradient clipping.
            alpha (float): Weight for the subspace sampling.
            subspace_dim (int): Dimension of the subspace.
            device (torch.device): Device to run the attack on.
        """
        super().__init__()
        assert targeted in ['targeted', 'untargeted']
        self.model = model
        self.sigma = sigma
        self.targeted = targeted
        self.grad_clip_threshold = grad_clip_threshold
        self.subspace_dim = subspace_dim
        self.device = device
        self.attack_loss = attack_loss
        self.penalty_fun = penalty_fun
        self.penalty_eta = penalty_eta
        
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
        initial_img: torch.Tensor,
        target_labels: torch.Tensor,
        subspace_estimation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the gradient of the loss function using zeroth order optimization.

        Args:
            evaluate_img (torch.Tensor): Input image to be attacked.
            target_labels (torch.Tensor): Target labels for targeted attack, or ground truth labels for untargeted attack.
            subspace_estimation (bool, optional): Whether to use subspace estimation. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Estimated gradient and loss.
        """
        # exceed the max sample will cause OOM
        grad_accumulation_steps = self.sample_num // self.max_sample_num_per_forward
        total_grads = []
        total_loss = []

        for _ in range(grad_accumulation_steps):
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
                assert self.subspace_dim is not None, 'subspace_dim must be specified when using subspace estimation'
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
            penalty_loss = self.penalty_fun(evaluate_imgs, initial_img, self.penalty_eta)
            loss = self.attack_loss(prediction, target_labels, self.targeted) + penalty_loss

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
        return total_grads, total_loss.detach(), penalty_loss

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
        """
        Compute the true loss of the input image.

        Args:
            evaluate_img (torch.Tensor): Input image to be evaluated.
            target_labels (torch.Tensor): Target labels for targeted attack, or ground truth labels for untargeted attack.

        Returns:
            torch.Tensor: True loss of the input image.
        """
        score = self.model(evaluate_img)
        loss = self.attack_loss(score, target_labels, self.targeted)
        return torch.squeeze(loss, 0)
