import torch
from torch import nn
from typing import Callable, Tuple
from .estimator import MetaZOEstimator
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import functional as F


class MetaTrainer:
    """
    A class for training a meta-learning based zeroth-order optimization algorithm.

    Args:
    - model (nn.Module): The neural network model to be trained.
    - attack_loss (Callable): The loss function used to evaluate the adversarial examples.
    - zo_estimator (MetaZOEstimator): The zeroth-order optimizer estimator.
    - device (torch.device): The device to run the training on.
    - targeted (str): The type of attack, either 'targeted' or 'untargeted'.

    Methods:
    - train(train_loader, test_loader, epoch_num, train_update_steps, tbptt_steps, test_update_steps, skip_query_rnn, skip_update_rnn, lr, checkpoint_path, checkpoint_interval): Trains the model for a given number of epochs.
    - _train_one_epoch(train_loader, update_steps, tbptt_steps, skip_query_rnn, skip_update_rnn, lr): Trains the model for one epoch.
    - _test_one_epoch(test_loader, update_steps, skip_query_rnn, skip_update_rnn, lr): Tests the model for one epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        zo_estimator: MetaZOEstimator,
        device: torch.device,
        targeted: str,
        attack_loss_fun: callable,
        penalty_fun: callable,
        penalty_eta: float = 0.1,
        max_sample_num_per_forward: int = 128,
        num_classes: int = 10,
    ) -> None:
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            attack_loss (Callable): The loss function to be used for adversarial attacks.
            zo_estimator (MetaZOEstimator): The zeroth-order estimator to be used for optimization.
            device (torch.device): The device on which to run the training.
            targeted (str): Whether the attack is targeted or untargeted. Must be either 'targeted' or 'untargeted'.
        """
        assert targeted in [
            'targeted', 'untargeted'], "targeted must be either 'targeted' or 'untargeted'"
        zo_estimator = zo_estimator.to(device)
        self.model = model.eval().to(device)
        self.attack_loss_fun = attack_loss_fun
        self.penalty_fun = penalty_fun
        self.penalty_eta = penalty_eta
        self.zo_estimator = zo_estimator
        self.device = device
        self.targeted = targeted
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.max_sample_num_per_forward = max_sample_num_per_forward
        self.num_classes = num_classes

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epoch_num: int,
        train_update_steps: int,
        tbptt_steps: int,
        test_update_steps: int,
        skip_query_rnn: bool,
        skip_update_rnn: bool,
        lr: float,
        checkpoint_path: str,
        checkpoint_interval: int,
    ):
        """
        Trains the zeroth-order optimizer (ZO) estimator using the given training and test data loaders for the specified number of epochs.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            test_loader (DataLoader): The data loader for the test set.
            epoch_num (int): The number of epochs to train for.
            train_update_steps (DataLoader): The data loader for the training set update steps.
            tbptt_steps (int): The number of truncated backpropagation through time (TBPTT) steps.
            test_update_steps (DataLoader): The data loader for the test set update steps.
            skip_query_rnn (bool): Whether to skip the query RNN.
            skip_update_rnn (bool): Whether to skip the update RNN.
            lr (float): The learning rate.
            checkpoint_path (str): The path to save the checkpoint.
            checkpoint_interval (int): The interval at which to save the checkpoint.

        Returns:
            None
        """
        assert Path(checkpoint_path).exists(), "checkpoint_path must exist"
        min_test_loss = float('inf')
        for epoch in range(epoch_num):
            decrease_in_loss, final_loss = self._train_one_epoch(
                train_loader=train_loader,
                update_steps=train_update_steps,
                tbptt_steps=tbptt_steps,
                skip_query_rnn=skip_query_rnn,
                skip_update_rnn=skip_update_rnn,
                lr=lr
            )
            print(
                f"Epoch: {epoch}, Decrease in Loss: {decrease_in_loss:.4f}, Final Loss: {final_loss:.4f}")

            test_loss, loss_ratio = self._test_one_epoch(
                test_loader=test_loader,
                update_steps=test_update_steps,
                skip_query_rnn=skip_query_rnn,
                skip_update_rnn=skip_update_rnn,
                lr=lr
            )
            print(
                f"Epoch: {epoch}, Test Loss: {test_loss}, Loss Ratio: {loss_ratio}")

            if epoch % checkpoint_interval == 0:
                self.zo_estimator.save(
                    epoch=epoch, checkpoint_path=checkpoint_path, best=False)

            if test_loss < min_test_loss:
                min_test_loss = test_loss
                self.zo_estimator.save(
                    epoch=epoch, checkpoint_path=checkpoint_path, best=True)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        update_steps: int,
        tbptt_steps: int,
        skip_query_rnn: bool,
        skip_update_rnn: bool,
        lr: float,
    ):
        decrease_in_loss = 0.0
        final_loss = 0.0
        self.zo_estimator.train()

        initial_img, target = next(iter(train_loader))
        initial_img, target = initial_img.to(
            self.device), target.to(self.device)
        onehot_target = F.one_hot(target, num_classes=self.num_classes)
        adversarial_img = initial_img.detach().clone().requires_grad_(True)
        output = self.model(adversarial_img)

        attack_loss = self.attack_loss_fun(
            output, onehot_target, self.targeted).item()
        penalty_loss = self.penalty_fun(
            adversarial_img, initial_img, self.penalty_eta)
        initial_loss = attack_loss + penalty_loss
        print("\ninitial loss: ", initial_loss.item())

        for k in range(update_steps // tbptt_steps):
            self._reset_state(adversarial_img, k)
            loss_sum = 0

            for j in range(tbptt_steps):
                delta, previous_loss, regularize_loss = self.zo_estimator.zo_estimation(
                    model=self.model,
                    evaluate_img=adversarial_img,
                    initial_img=initial_img,
                    target_labels=onehot_target,
                    skip_query_rnn=skip_query_rnn,
                    skip_update_rnn=skip_update_rnn,
                    lr=lr
                )
                previous_loss = previous_loss.detach()
                delta = delta.reshape_as(adversarial_img)
                adversarial_img = adversarial_img + delta
                adversarial_img = adversarial_img.clamp(-0.5, 0.5)
                output = self.model(adversarial_img)
                loss = self.attack_loss_fun(
                    output, onehot_target, self.targeted)
                loss = loss + \
                    self.penalty_fun(
                        adversarial_img, initial_img, self.penalty_eta)

                loss_sum += (k * tbptt_steps + j) * (loss - previous_loss)
                loss_sum += regularize_loss

            self.zo_estimator.zero_grad()
            self.optimizer.zero_grad()
            loss_sum.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(
                self.zo_estimator.parameters(), 1)
            self.optimizer.step()

        decrease_in_loss = decrease_in_loss + (loss.item() / initial_loss)
        final_loss = final_loss + loss.item()

        return decrease_in_loss, final_loss

    def _reset_state(self, adversial_img, k):
        """
        Resets the state of the trainer.

        Parameters:
        - adversial_img: The adversarial image.
        - k: The current tbptt step number

        Returns:
        None
        """
        self.zo_estimator.reset_state(keep_states=k > 0)
        self.model.zero_grad()
        adversial_img.detach_()

    def _test_one_epoch(
        self,
        test_loader: DataLoader,
        update_steps: int,
        skip_query_rnn: bool,
        skip_update_rnn: bool,
        lr: float,
        num_test_samples: int = 10,
    ):
        self.zo_estimator.eval()
        loss_sum = 0
        loss_ratio = 0.0

        for i, (initial_img, target) in enumerate(test_loader):
            if i >= num_test_samples:
                break

            initial_img, target = initial_img.to(
                self.device), target.to(self.device)
            output = self.model(initial_img)
            onehot_target = F.one_hot(target, num_classes=self.num_classes)
            adversial_img = initial_img.clone().detach().requires_grad_(False)
            attack_loss = self.attack_loss_fun(
                output, onehot_target, self.targeted).item()
            penalty_loss = self.penalty_fun(
                adversial_img, initial_img, self.penalty_eta)
            init_loss = attack_loss + penalty_loss
            self.zo_estimator.reset_state(keep_states=False)

            for k in range(update_steps):
                delta, test_loss, _ = self.zo_estimator.zo_estimation(
                    model=self.model,
                    evaluate_img=adversial_img,
                    initial_img=initial_img,
                    target_labels=onehot_target,
                    skip_query_rnn=skip_query_rnn,
                    skip_update_rnn=skip_update_rnn,
                    lr=lr
                )
                test_loss = test_loss.detach()
                delta = delta.reshape_as(adversial_img)
                adversial_img = adversial_img + delta
                adversial_img = torch.clip(adversial_img, -0.5, 0.5)

            # use last loss as the final loss
            loss_sum += test_loss.item()
            loss_ratio += test_loss.item() / init_loss

        return loss_sum / num_test_samples, loss_ratio / num_test_samples
