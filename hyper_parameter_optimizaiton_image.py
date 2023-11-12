from attack_mutilgpu import esal
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call
import optuna
from optuna.visualization.matplotlib import plot_param_importances, plot_pareto_front, plot_parallel_coordinate
from functools import partial
from matplotlib import pyplot as plt

def target(trail: optuna.Trial, cfg):
    samples_per_draw = trail.suggest_int('sample_per_draw', 2, 100, 2)
    max_learning_rate = trail.suggest_float('max_lr', 0.0001, 5, log=True)
    k_init = trail.suggest_int('k_init', 1, 90)
    cfg.setup.algorithm.samples_per_draw = samples_per_draw
    cfg.setup.optimizer.max_lr =  max_learning_rate
    cfg.setup.algorithm.k = k_init
    images, labels, model = call(cfg.dataset_and_model.dataset_and_model)
    result = esal(images=images, labels=labels, model=model, cfg=cfg.setup)
    query = result['query']
    query = sum(query) / len(query)
    acc = result['acc']
    return acc, query

@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg: DictConfig):
    device = cfg.setup.general_setup.visible_device
    # print("============ Information: ============")
    # print('GPU device: ', device)
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    target_with_cfg = partial(target, cfg=cfg)
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(target_with_cfg, n_trials=50)
    best_trails = study.best_trials
    print('parameters of parento front')
    dataset = "graph/" + str(cfg.setup.algorithm.targeted) + '/' + str(cfg.setup.algorithm.dataset_name) + '/'
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    for best_trail in best_trails:
        print('params:', best_trail.params)

    fig = plot_pareto_front(study, target_names=['acc', 'query'])
    plt.savefig(dataset + 'parento.jpg')
    fig = plot_param_importances(study, target=lambda t: t.values[0])
    plt.savefig(dataset + 'acc_importance.jpg')
    fig = plot_param_importances(study, target=lambda t: t.values[1])
    plt.savefig(dataset + 'query_importance.jpg')
    fig = plot_parallel_coordinate(study, target=lambda t: t.values[0])
    plt.savefig(dataset + 'acc_cord.jpg')
    fig = plot_parallel_coordinate(study, target=lambda t: t.values[1])
    plt.savefig(dataset + 'query_cord.jpg')
    
    

if __name__ == '__main__':
    main()