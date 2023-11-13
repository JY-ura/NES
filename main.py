from statistics import mean, median
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
from attack import esal
import csv
import os
import time
import torch
import wandb


def print_info(result: dict):
    print("####################################### Result #######################################")
    print("acc: {:.4f}".format(result['acc']))
    print("query_list: ", result['query'])
    print("query_mean:", mean(result['query']))
    print("query_median:", median(result['query']))
    print("l0_norm: {:.2f}".format(result['l0']))
    print("l2 distance: {:.2f}".format(result['l2']))
    print("linf distance: {:.2f}".format(result['linf']))

    # print("psnr:{:.2f}".format(mean(result['psnr'])))
    # print("ssim:{:.2f}".format(mean(result['ssim'])))


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg: DictConfig):
    if cfg.wandb:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                project=cfg.project, entity=cfg.entity, name=cfg.name)
        wandb.login()
    
    images, labels, model = call(cfg.dataset_and_model.dataset_and_model)
    start = time.time()
    result = esal(images, labels, model, cfg)
    end = time.time()
    print("It took", (end-start)/60, "minutes.")
    print_info(result)

    if cfg.save_result:
        save_path = f"log/{cfg.setup.algorithm.dataset_name}/{cfg.setup.algorithm.targeted}/{cfg.setup.algorithm.if_overlap}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # data=[cfg.setup.algorithm.k, cfg.setup.algorithm.epsilon, cfg.setup.optimizer.max_lr,
        #     '{:.2%}'.format(acc),'{:.2f}'.format(mean(query_list)), median(query_list), '{:.2f}'.format(l0), '{:.2f}'.format(l2), '{:.2f}'.format(linf)]
        data = [cfg.setup.algorithm.filterSize, '{:.2%}'.format(result['acc']),
                '{:.2f}'.format(mean(result['query'])),
                '{:.2f}'.format(median(result['query'])),
                '{:.2f}'.format(mean(result['psnr'])),
                '{:.2f}'.format(mean(result['ssim'])),
                '{:.2f}'.format(result['l0']),
                '{:.2f}'.format(result['l2']),
                '{:.2f}'.format(result['linf'])
                ]
        with open(save_path+"result.csv", 'a+', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data)


if __name__ == '__main__':
    main()
