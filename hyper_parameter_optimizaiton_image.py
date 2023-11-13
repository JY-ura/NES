from attack import esal
from statistics import mean, median
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call
import optuna
from optuna.visualization.matplotlib import plot_param_importances, plot_pareto_front, plot_parallel_coordinate
from functools import partial
from matplotlib import pyplot as plt
import yaml

def print_info(result: dict):
    print("####################################### Result #######################################")
    print("acc: {:.4f}".format(result['acc']))
    print("query_list: ", result['query'])
    print("query_mean:", mean(result['query']))
    print("query_median:", median(result['query']))
    print("l0_norm: {:.2f}".format(result['l0']))
    print("l2 distance: {:.2f}".format(result['l2']))
    print("linf distance: {:.2f}".format(result['linf']))


def target(trail: optuna.Trial, cfg):
    samples_per_draw = trail.suggest_int('sample_per_draw', 2, 100, 2)
    max_learning_rate = trail.suggest_float('max_lr', 1, 100, log=True)
    k_init = trail.suggest_int('k_init', 1, 20)
    eta = trail.suggest_float('eta', 0.01, 5)

    cfg.setup.algorithm.samples_per_draw = samples_per_draw
    cfg.scheduler.max_lr = max_learning_rate
    cfg.setup.algorithm.k = k_init
    cfg.dataset_and_model.eta = eta

    images, labels, model = call(cfg.dataset_and_model.dataset_and_model)
    result = esal(images=images, labels=labels, model=model, cfg=cfg)
    query = result['query']
    query = sum(query) / len(query)
    acc = result['acc']
    print_info(result)
    return acc, query


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg: DictConfig):

    n_trials = 50

    target_with_cfg = partial(target, cfg=cfg)
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(target_with_cfg, n_trials=n_trials)
    best_trails = study.best_trials
    print('parameters of parento front')
    dataset = "graph/" + str(cfg.targeted) + \
        '/' + str(cfg.dataset_and_model.name) + '/' + str(cfg.dataset_and_model.dataset_and_model.model_type) + '/'
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    for best_trail in best_trails:
        print('params:', best_trail.params)

    fig = plot_pareto_front(study, target_names=['acc', 'query'])
    name = str(cfg.setup.algorithm.perturb_rate) + '_' + str(cfg.setup.algorithm.if_overlap) + '_' + str(n_trials) 

    plt.savefig(dataset + name + '_parento.jpg')
    fig = plot_param_importances(study, target=lambda t: t.values[0])
    plt.savefig(dataset + name + '_acc_importance.jpg')
    fig = plot_param_importances(study, target=lambda t: t.values[1])
    plt.savefig(dataset + name + '_query_importance.jpg')
    fig = plot_parallel_coordinate(study, target=lambda t: t.values[0], )
    plt.savefig(dataset + name + '_acc_cord.jpg')
    fig = plot_parallel_coordinate(study, target=lambda t: t.values[1], )
    plt.savefig(dataset + name + '_query_cord.jpg')


if __name__ == '__main__':
    main()
