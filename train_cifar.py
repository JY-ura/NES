
import os
os.environ['WANDB_MODE'] = 'offline'
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torchvision
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader, random_split
from dataset_and_model.cifar import *
import numpy as np
import datetime
from ignite.utils import setup_logger
from ignite.handlers import create_lr_scheduler_with_warmup, Checkpoint, DiskSaver, global_step_from_engine
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite import metrics
from ignite.contrib.handlers import wandb_logger
from pathlib import Path
from typing import *
from torch.cuda.amp import grad_scaler
from ignite.engine import Engine
from d2l import torch as d2l


def get_dataset(data_path, num_pic=10000):
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True, download=True)
    train_data, val_data = random_split(data, [0.9,0.1], torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)
    
    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False, download=True,)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def get_trainer(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]],
        amp_model: bool,
        scaler: Optional[grad_scaler.GradScaler],
        output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x,y, y_pred, loss : loss.item()
)->Engine:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if amp_model:
        assert scaler is not None, "scaler must be setted if amp is True"
    
    def update(engine: Engine, batch: Sequence):
        model.train()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast(amp_model):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return output_transform(x,y,y_pred, loss)
    return Engine(update)

def log_metrics(logger, epoch, elapsed, tag, metrics):
    """
    Logs the evaluation metrics for a given epoch and tag.

    Args:
        logger: The logger object to use for logging.
        epoch (int): The epoch number.
        elapsed (float): The time taken for evaluation.
        tag (str): The tag for the evaluation metrics.
        metrics (dict): A dictionary containing the evaluation metrics.

    Returns:
        None
    """
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )



@hydra.main(config_path='config', config_name='train_cifar', version_base=None)
def main(cfg: DictConfig):
    # print(cfg)
    wandb.init(
        project=cfg.project,
        name=f'{cfg.dataset_and_model.dataset_and_model.model_type}-lr{cfg.lr}',
        entity=cfg.entity,
        job_type='train_cifar10_resnet',
        config=OmegaConf.to_container(cfg=cfg, resolve=True)
    )

    device = torch.device(
        f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else 'cpu'
    )
    
    train_loader, val_loader, _ = get_dataset(
        data_path='./datasets/cifar')
    
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset_and_model.num_labels)
    nn.init.xavier_uniform_(model.fc.weight)
    model.to(device)

    train(
        train_loader,
        val_loader,
        model,
        device,
        cfg = wandb.run.config
    )

def train(train_loader,val_loader, model, device, cfg):
    
    main_logger = setup_logger(name='main')
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.1
    ).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=2e-5)
    para_1x = [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    optimizer = torch.optim.SGD([{'params': para_1x}, {'params': model.fc.parameters(), 'lr': cfg.lr * 10}],
                                    lr=cfg.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.T_max)
    scheduler = create_lr_scheduler_with_warmup(
        lr_scheduler=scheduler, 
        warmup_start_value=1e-5,
        warmup_end_value=cfg.lr,
        warmup_duration=cfg.warmup_epoch
    )
    amp_mode = 'amp' if cfg['use_amp'] else None
    scaler= torch.cuda.amp.GradScaler(
        enabled=cfg.use_amp 
    ) if cfg.use_amp else None
    if scaler is not False:
        assert amp_mode is not None, "use_amp must be True if scaler is setted"

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        amp_mode='amp',
        scaler=scaler,
        )
    
    loss = metrics.Loss(loss_fn= loss_fn, device=device)
    acc =  metrics.Accuracy(device=device)
    top_5_acc = metrics. TopKCategoricalAccuracy(k=5, device=device)

    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_lr(engine: Engine):
        print(f'lr:{optimizer.param_groups[0]["lr"]}')
    
    train_evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'loss': loss,
            'top1 acc': acc,
            'top5 acc': top_5_acc
        },
        device=device,
        amp_mode=True if amp_mode == 'amp'  else False
    )

    val_evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'loss': loss,
            'top1 acc': acc,
            'top5 acc': top_5_acc
        },
        device=device,
        amp_mode=True if amp_mode == 'amp'  else False
    )

    trainer.logger = setup_logger('Main Train')
    train_evaluator.logger = setup_logger('tarin evaluator')
    val_evaluator.logger = setup_logger('val evaluatior')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results_val(engine: Engine):
        epoch = trainer.state.epoch
        state = val_evaluator.run(val_loader)
        log_metrics(main_logger, epoch, state.times['COMPLETED'], 'Valdation', state.metrics)
    

    @trainer.on(Events.EPOCH_COMPLETED(every=4))
    def log_train_results_train(engine: Engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(main_logger, epoch, state.times['COMPLETED'], 'Training', state.metrics)


    train_logger = wandb_logger.WandBLogger()
    train_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=cfg.report_every),
        tag = 'training',
        output_transform= lambda loss: {'train loss': loss},
        global_step_transform = lambda *_: trainer.state.iteration,
    )

    for engine, name in zip([train_evaluator, val_evaluator], ['trianing', 'validation']):
        logger = wandb_logger.WandBLogger()
        logger.attach_output_handler(
            engine,
            event_name=Events.COMPLETED,
            tag = name,
            metric_names = ['loss', 'top1 acc', 'top5 acc'],
            global_step_transform = lambda *_: trainer.state.iteration,
        )

    date = datetime.datetime.now()
    dir_name = f"./model_files/{cfg.dataset_and_model['dataset_and_model']['model_type']}/{cfg.dataset_and_model['name']}/{date.year}.{date.month}.{date.day}.{date.hour}.{date.minute}.{date.second}"
    checkpoint_handler = Checkpoint(
        to_save={'model':model},
        save_handler=DiskSaver(dirname=dir_name, require_empty=False),
        filename_prefix='best',
        n_saved=2,
        score_function=lambda engine: engine.state.metrics['top1 acc'],
        score_name='top1_acc',
        global_step_transform=global_step_from_engine(trainer)
    )

    val_evaluator.add_event_handler(
        Events.COMPLETED, checkpoint_handler
    )

    @trainer.on(Events.COMPLETED)
    def upload_training(engine: Engine):
        artifact = wandb.Artifact(
            name='model_and_training_status',
            type='model',
            description='train dnn network',
            metadata={
                'dataset': cfg.dataset_and_model.name,
                'batch_size': cfg.batchsize,
                'lr': cfg.lr
            }
        )

        for file in Path(dir_name).iterdir():
            artifact.add_file(str(file), name=file.name)

        print("tring to upload artifact ...")
        wandb.run.log_artifact(artifact)
        print("upload artifact successfully")

    trainer.run(train_loader, max_epochs=cfg.epoch)
    train_logger.close()
    wandb.finish()


if __name__=="__main__":
    main()