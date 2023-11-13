import torch
from typing import Tuple
import torch.nn as nn
from optimizer import margin_loss

# compute the true gradient


def compute_loss(evaluate_img: torch.Tensor, target_labels: torch.Tensor, model, targeted,):
    score = model(evaluate_img)
    loss = margin_loss(score, target_labels, targeted)
    return torch.squeeze(loss, 0)

loss_fn = margin_loss
# @torch.no_grad()


def get_grad_estimation(
    model: nn.Module,
    evaluate_img: torch.Tensor,
    target_labels: torch.Tensor,
    sample_per_draw: int,
    batch_size: int,
    sigma: float,
    targeted: bool,
    host,
    norm_theshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """get gradient estimation on evaluate_img using nes

    Args:
        evaluate_img (torch.Tensor): _description_
        target_label (torch.Tensor): replicated target labels for computing loss
        sample_per_draw (int): _description_
        batch_size (int): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    total_batch_num = sample_per_draw // batch_size
    total_grads = []
    total_loss = []
    total_grad_norm = []
    for _ in range(total_batch_num):
        noise = torch.normal(mean=0.0, std=1.0, size=(
            batch_size // 2,) + evaluate_img.shape[1:], device=host)
        # generate + delta and - delta for evaluation
        noise = torch.concat([noise, -noise], dim=0)
        evaluate_imgs = evaluate_img + noise * sigma

        score = model(evaluate_imgs)
        loss = loss_fn(score, target_labels, targeted)

        loss = loss.reshape(-1, 1, 1, 1).repeat(evaluate_img.shape)
        grad = loss * noise / sigma / 2

        total_grads.append(torch.mean(grad, dim=0, keepdim=True))
        total_loss.append(torch.mean(loss))
        # torch.cuda.empty_cache()

    total_grads = torch.mean(torch.concat(total_grads), dim=0, keepdim=True)
    if torch.norm(total_grads, p=2) > norm_theshold:
        total_grads = total_grads/torch.norm(total_grads, p=2)*norm_theshold
    total_loss = torch.mean(torch.tensor(total_loss, device=host), dim=0)
    return total_grads, total_loss, 
