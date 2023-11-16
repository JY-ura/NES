from l2l_zo.trainer import MetaTrainer
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import TensorDataset, DataLoader
from optimizer import margin_loss, cross_entorpy_loss
from l2l_zo.estimator import MetaZOEstimator
import os

def get_model(dataset, path):
    if dataset == 'cifar10':
        from dataset_and_model.cifar import CIFARModelTorch
        model = CIFARModelTorch()
        model.load_state_dict(torch.load(path))

        return model


def get_dataset(dataset, path):
    if dataset == 'cifar10':
        from dataset_and_model.cifar import CIFAR
        cifar_dataset = CIFAR(data_path=path, num_pic=10000)
        train_img, train_lab = cifar_dataset.train_data, cifar_dataset.train_labels
        val_img, val_lab = cifar_dataset.validation_data, cifar_dataset.validation_labels
        
        train_img_tensor = torch.tensor(train_img, )
        train_lab_tensor = torch.argmax(torch.tensor(train_lab, ), dim=-1)
        val_img_tensor = torch.tensor(val_img, )
        val_lab_tensor = torch.argmax(torch.tensor(val_lab,), dim=-1)

        batch_size = 1
        train_set = TensorDataset(train_img_tensor, train_lab_tensor)
        val_set = TensorDataset(val_img_tensor, val_lab_tensor)

        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)

        return train_loader, val_loader


@hydra.main(config_name='train_rnn', config_path='config', version_base=None)
def main(cfg: DictConfig):
    device = torch.device(
        f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else 'cpu')

    # 1. Load the model
    model = get_model(dataset=cfg.dataset_and_model.name, path = cfg.dataset_and_model.dataset_and_model.model_path)
    model.to(device)
    # 2. Load the dataset
    train_loader, val_loader = get_dataset(dataset=cfg.dataset_and_model.name, path = cfg.dataset_and_model.dataset_and_model.data_path)

    # 3. initial the zo_estimator
    max_batch_forward = cfg.max_batch_forward
    zo_estimator = MetaZOEstimator(
        sample_num=max_batch_forward,
        max_sample_num_per_forward=max_batch_forward,
        targeted=cfg.targeted,
        grad_clip_threshold=cfg.dataset_and_model.grad_norm_threshold,
        device=device,
        flattened_input_dim=cfg.dataset_and_model.num_channels * cfg.dataset_and_model.image_size * cfg.dataset_and_model.image_size,
    )

    # 4. initial the update rnn trainer
    trainer = MetaTrainer(
        model=model,
        attack_loss=cross_entorpy_loss,
        zo_estimator=zo_estimator,
        device=device,
        targeted=cfg.targeted,
        max_sample_num_per_forward=max_batch_forward,
        num_classes=cfg.dataset_and_model.num_classes,
    )

    checkpoint_path_update_rnn = 'model_files/l2lzo/cifar10/update_rnn/'
    if not os.path.exists(checkpoint_path_update_rnn):
        os.makedirs(checkpoint_path_update_rnn)

    trainer.train(
        train_loader=train_loader,
        test_loader=val_loader,
        epoch_num=cfg.epoch,
        train_update_steps=cfg.train_update_steps,
        test_update_steps=cfg.test_update_steps,
        tbptt_steps=cfg.tbptt_steps,
        skip_query_rnn=True,
        skip_update_rnn=False,
        lr=cfg.lr,
        checkpoint_path=checkpoint_path_update_rnn,
        checkpoint_interval=cfg.checkpoint_interval,
    )

    checkpoint_path_query_rnn = 'model_files/l2lzo/cifar10/query_rnn/'
    if not os.path.exists(checkpoint_path_query_rnn):
        os.makedirs(checkpoint_path_query_rnn)

    trainer.train(
        train_loader=train_loader,
        test_loader=val_loader,
        epoch_num=cfg.epochs,
        train_update_steps=cfg.train_update_steps,
        test_update_steps=cfg.test_update_steps,
        tbptt_steps=cfg.tbptt_steps,
        skip_query_rnn=False,
        skip_update_rnn=True,
        lr=cfg.lr,
        checkpoint_path=checkpoint_path_query_rnn,
        checkpoint_interval=cfg.checkpoint_interval,
    )

    


if __name__ == '__main__':
    main()