import torch
from typing import Tuple
import torch.nn as nn
from optimizer import margin_loss

class GradEstimation:
    def __init__(self, model, sample_per_draw, batch_size, sigma, targeted, norm_threshold, alpha, pre_k_grad, host) -> None:
        self.model = model
        self.sample_per_draw = sample_per_draw
        self.batch_size = batch_size
        self.sigma = sigma
        self.targeted = targeted
        self.norm_threshold = norm_threshold
        self.alpha = alpha
        self.pre_k_grad = pre_k_grad
        self.host = host
        self.sigma_sqrt = 0.1
        self.grads = None
        self.pre_grad_list = []
        self.loss_func = margin_loss

    def get_grad_estimation(
        self,
        evaluate_img: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        total_batch_num = self.sample_per_draw // self.batch_size
        total_grads = []
        total_loss = []
        for _ in range(total_batch_num):
            noise = torch.normal(mean=0.0, std=1.0, size=(
                self.batch_size // 2,) + evaluate_img.shape[1:], device=self.host)

            if self.grads is not None:
                self.pre_grad_list.append(self.grads.flatten())
                if len(self.pre_grad_list) == self.pre_k_grad:
                    pre_grad = torch.stack(
                        self.pre_grad_list, dim=0).to(self.host)
                    noise = self.sample(
                        pre_grad.transpose(0, 1),
                        evaluate_img.flatten().shape[0]
                    )
                    noise = noise.view((noise.size(0),) +
                                       evaluate_img[0].size())
                    self.pre_grad_list.pop(0)

            # generate + delta and - delta for evaluation
            noise = torch.concat([noise, -noise], dim=0)

            evaluate_imgs = evaluate_img + noise * self.sigma

            score = self.model(evaluate_imgs)
            loss = self.loss_fn(score, target_labels, self.targeted)

            loss = loss.reshape(-1, 1, 1, 1).repeat(evaluate_img.shape)
            grad = loss * noise / self.sigma / 2

            total_grads.append(torch.mean(grad, dim=0, keepdim=True))
            total_loss.append(torch.mean(loss))
            # torch.cuda.empty_cache()

        total_grads = torch.mean(torch.concat(
            total_grads), dim=0, keepdim=True)
        if torch.norm(total_grads, p=2) > self.norm_theshold:
            total_grads = total_grads / \
                torch.norm(total_grads, p=2)*self.norm_theshold
        total_loss = torch.mean(torch.tensor(
            total_loss, device=self.host), dim=0)
        return total_grads, total_loss

    def sample(self, grads, d):
        noise_full = torch.rand(self.batch_size//2, d, 1).to(self.host)
        nosie_subspace = torch.rand(
            self.batch_size//2, self.pre_k_grad, 1).to(self.host)

        q, _ = torch.qr(grads)
        noise = self.sigma_sqrt * torch.sqrt(torch.tensor(1-self.alpha / d * 0.1)) * noise_full \
            + self.sigma_sqrt * torch.sqrt(torch.tensor((self.alpha) /
                                                        self.pre_k_grad * 1.0)) * torch.matmul(q, nosie_subspace)
        return noise.squeeze(-1)
    

    def compute_true_loss(self, evaluate_img: torch.Tensor, target_labels: torch.Tensor)->torch.Tensor:
        score = self.model(evaluate_img)
        loss = self.loss_func(score, target_labels, self.targeted)
        return torch.squeeze(loss, 0)
