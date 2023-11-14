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


def sample(grads, d, host, batch, pre_k_grad=3, alpha=0.5, sigma_sqrt=0.1):
    noise_full = torch.rand(batch//2, d, 1).to(host)
    nosie_subspace = torch.rand(batch//2, pre_k_grad, 1).to(host)

    q, _ = torch.qr(grads)
    noise = sigma_sqrt * torch.sqrt(torch.tensor(1-alpha / d * 0.1)) * noise_full \
        + sigma_sqrt * torch.sqrt(torch.tensor((alpha) /
                                  pre_k_grad * 1.0)) * torch.matmul(q, nosie_subspace)
    return noise.squeeze(-1)


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
    grads,
    pre_grad_list,
    pre_k_grad,
    alpha
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient estimation on evaluate_img using NES.

    Args:
        model (nn.Module): PyTorch model to compute gradients for.
        evaluate_img (torch.Tensor): Input image to compute gradients for.
        target_labels (torch.Tensor): Target labels for computing loss.
        sample_per_draw (int): Number of samples to draw for each batch.
        batch_size (int): Batch size.
        sigma (float): Standard deviation of the noise distribution.
        targeted (bool): Whether to perform targeted attack or not.
        host: Device to perform computation on.
        norm_theshold (float): Threshold for gradient norm.
        grads: Previous gradients.
        pre_grad_list: List of previous gradients.
        pre_k_grad: Number of previous gradients to use.
        alpha: Learning rate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the computed gradients and loss.
    """
    total_batch_num = sample_per_draw // batch_size
    total_grads = []
    total_loss = []
    for _ in range(total_batch_num):
        noise = torch.normal(mean=0.0, std=1.0, size=(
            batch_size // 2,) + evaluate_img.shape[1:], device=host)

        if grads is not None:
            pre_grad_list.append(grads.flatten())
            if len(pre_grad_list) == pre_k_grad:
                pre_grad = torch.stack(pre_grad_list, dim=0).to(host)
                noise = sample(pre_grad.transpose(0, 1), evaluate_img.flatten(
                    ).shape[0], host, batch_size, pre_k_grad=pre_k_grad, alpha=alpha)
                noise = noise.view((noise.size(0),) + evaluate_img[0].size())
                pre_grad_list.pop(0)

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
