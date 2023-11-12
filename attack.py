from statistics import median, mean
import torch
from statistics import mean, median
import numpy as np
from omegaconf import DictConfig
from optimizer import Momentum, cos_scheduler, margin_loss, cross_entorpy_loss, Adam, step_lr_scheduler
from typing import Callable
from typing import Tuple
from torch.nn import functional as F
from utils.general_utils import *
from torch.nn import DataParallel
import os
import shutil
from utils.get_grad import *
from utils.select_groups import *

loss_fn = margin_loss
SEED = 0
torch.random.initial_seed()
torch.random.manual_seed(0)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)



def remove_error(idx):
    # idx = torch.tensor(idx)
    # os 读取目录下的文件，默认排序 为乱序！！！
    filelist = os.listdir(
        '/media/mllab/yym/code/2/nips_FTAS/NES/datasets/imagenet/val')

    filelist.sort()
    idx_list = idx[0].tolist()
    namelist = []
    for i in idx_list:
        namelist.append(filelist[i])

    for name in namelist:
        # print(filelist[i])
        shutil.move("datasets/imagenet/val/" + name, "datasets/imagenet/wrong")
    # print(filelist)


def image_save(adv_image, image, orig_label, target_label, dataset, targeted, index, grouping_strategy):
    """save image to local

    Args:
        adv_image (_type_): the adversarial image
        image (_type_): the orignal image
        orig_label (_type_): the original label
        target_label (_type_): the target label for target attack, the original label for untarget attack
        dataset (_type_): "cifar, mnist, imagenet"
        targeted (_type_): if targeted
        index (_type_): _description_
        grouping_strategy (_type_): _description_
    """
    adv_image = img_transform(adv_image.cpu().numpy())
    ori_image = img_transform(image.cpu().numpy())
    orgi_ = orig_label.cpu()
    img_save(ori_image, index, "origin", grouping_strategy,
             dataset, orgi_, target_label)
    img_save(adv_image, index, targeted + '_adv_',
             grouping_strategy, dataset, orgi_, target_label)
    img_save(np.abs(adv_image - ori_image), index, targeted+"_delta",
             grouping_strategy, dataset, orgi_, target_label)


def esal(
        images: np.ndarray, 
        labels: np.ndarray, 
        model: torch.nn.Module, 
        cfg: DictConfig, 
        norm_theshold=10):
    """perform esal_0+inf attack on target model

    Args:
        images (np.ndarray): input(clean) images
        labels (np.ndarray): corresponding labels
        model (torch.nn.Module): target model to attack
        cfg (DictConfig): setup for algorithm
        loss_fn (Callable): take in score and target label and return loss

    Return:
        results (Dict): the required results
        {'query' : list of all image queries,
         'acc'   : ASR,
         'l0'    : the l0 norm,
         'l2'    : the l2 norm,
         'linf'  : the linf norm,
         'psnr'  : psnr,
         'ssim'  : ssim}
    """

    targeted = cfg.targeted

    # algorithm
    alg = cfg.setup.algorithm
    sigma = alg.sigma  # refers to sigma in the paper
    samples_per_draw = alg.samples_per_draw
    batch_size = alg.batch_delta_size  # max batch size to evaluate the output
    plateu_length = alg.plateu_length
    grouping_strategy = alg.grouping_strategy
    if_overlap = alg.if_overlap
    epsilon = alg.epsilon  # correspond to epsilon in the paper
    perturb_rate = alg.perturb_rate
    k_init = alg.k
    # k_init = round(1.0*d.item()/filtersize.item()/filtersize.item()/channels*perturb_rate)
    print("k_init", k_init)

    # optimizer
    opti = cfg.setup.optimizer
    max_learning_rate = opti.max_lr
    min_learning_rate = opti.min_lr
    image_size = torch.tensor(cfg.dataset_and_model.image_size)  # image size
    num_labels = cfg.dataset_and_model.num_labels
    momentum = opti.momentum
    drop_epoch = opti.drop_epoch

    # dataset and model
    d_m = cfg.dataset_and_model
    channels = d_m.num_channels
    dataset = d_m.name
    model_name = d_m.dataset_and_model.model_type
    filtersize = torch.tensor(d_m.filterSize)
    stride = d_m.stride

    d = channels*image_size*image_size

    if dataset == 'imagenet':
        max_per_draw = 100
        if model_name == 'inceptionv3':
            image_size = torch.tensor(299)
            max_perturb = round(1.0*d.item()/filtersize.item() /
                                filtersize.item()/channels*perturb_rate)+20
        elif model_name == 'VT':
            image_size = torch.tensor(224)
            max_perturb = round(1.0*d.item()/filtersize.item() /
                                filtersize.item()/channels*perturb_rate)+20
    else:
        if alg.if_overlap == "overlapping":
            max_perturb = 16  # cifar mnist untarget: 16 target: 20
        else:
            max_perturb = 23  # cifar mnist untarget: 16 target: 20
        max_per_draw = 64

    parallel_device_num = cfg.parallel_device_num
    device_num = torch.cuda.device_count()
    parallel_device_num = min(device_num, parallel_device_num)
    device_ids = list(np.arange(parallel_device_num))

    if parallel_device_num > 1:
        print('using data parallel on', device_ids)
        model = DataParallel(model, device_ids)
        model = model.to(device_ids[0])
    else:
        model = model.to(device_ids[0])
        print('using single gpu')

    # test model acc, get correct index and attack correct images
    # print(f'image shape: {images.shape}')
    # print(f'label shape: {labels.shape}')

    images = torch.tensor(images).float().to(device_ids[0])
    labels = torch.tensor(labels).to(device_ids[0])

    if targeted == 'targeted':
        print(
            f"It performs targeted attack on {dataset} dataset, {model_name} model!")
        target_class = torch.tensor(
            [pseudorandom_target(index, num_labels, orig_label)
             for index, orig_label in enumerate(labels)], dtype=torch.int64
        ).to(device_ids[0])
    else:
        print(
            f"It performs untargeted attack on {dataset} dataset, {model_name} model!")
        target_class = labels

    host = torch.device(device_ids[0])

    # validation on model
    with torch.no_grad():
        score = model(images)
        pred = torch.argmax(score, dim=-1)
        torch.cuda.empty_cache()
        # remove_id = torch.where(pred != labels)
        # remove_error(remove_id)
        correct_idx = pred == labels

        acc = torch.mean(correct_idx.float())
    print(f'model acc: {acc.cpu()}')
    assert acc > 0.65, 'we s/hould a/ttack a well trained model'

    # get correct samples to attack
    images = images[correct_idx]
    labels = labels[correct_idx]
    target_class = target_class[correct_idx]
    print(f"Corrrectly classified images are {images.shape[0]}.")

    def attack(
            target_img: torch.Tensor, 
            target_label: torch.Tensor, 
            original_label: torch.Tensor,
            samples_per_draw, 
            batch_size, 
            masks: torch.Tensor, 
            max_learning_rate, 
            cfg):
        """perform easl attack on target img 

        Args:
            target_img (torch.Tensor): target attack img
            target_label (torch.Tensor): label for target attack
            original_label (torch.Tensor): original label for the image
            samples_per_draw (int): times of estimating gradient in each attacking
            batch_size (int):  The parallel dimensions of a picture
            index: Initial group index
        """
        with torch.no_grad():
            # initialization
            target_img = target_img.reshape((1,) + target_img.shape)
            gradient_transformer = Momentum(
                variabels=target_img, momentum=momentum)
            admm_transformer = Adam(target_img)
            scheduler = get_lr_scheduler(**cfg.scheduler)
            num_queries = 0
            delta = torch.rand(channels, image_size,
                               image_size).to(device_ids[0])
            # delta = torch.tensor(0).to(device_ids[0])
            flag = False
            lower_bond = torch.maximum(
                torch.tensor(-epsilon).to(device_ids[0]), -0.5 - target_img)
            uppder_bond = torch.minimum(torch.tensor(
                epsilon).to(device_ids[0]), 0.5 - target_img)
            adv_image = target_img.detach().clone()
            # target_labels = F.one_hot(target_label, num_classes=num_labels).repeat(batch_size,
            #                                                                        1)  # create one-hot target labels as (batch_size, num_class)

            last_ls = []
            k = k_init
            k_hat = k_init
            last_query = cfg.max_query

            # # initial attack
            # l0_norm = torch.sum((delta != 0).float())
            # l2_norm = torch.norm(delta)
            # linf_norm = torch.max(torch.abs(delta))
            # pro_delta = torch.clip(delta, lower_bond, uppder_bond)  # clip the delta to satisfy l_inf norm
            # h = pro_delta ** 2 - 2 * pro_delta * delta # shape: [1, image.shape]
            # unclip_delta = greedy_project(h, delta, masks.clone(), k)
            # delta = torch.clip(unclip_delta, lower_bond, uppder_bond)
            # adv_image = torch.clip(target_img.detach().clone() + delta, -0.5, 0.5)  # adversial samples

            # for iter in range(max_iters):
            for iters in range(cfg.max_query):

                # check if we can make an early stopping
                pred = torch.argmax(model(adv_image, ))
                if is_sucessful(target_label, pred):
                    flag = True
                    # print(f'[succ] Iter: {iters}, groups: {k}, query: {num_queries}, loss: {loss.cpu():.3f}, l0:{l0_norm.cpu():.0f}, l2:{l2_norm.cpu():.1f}, linf:{linf_norm.cpu():.2f}, prediction: {pred}, target_label:{target_label}')
                    break

                target_labels = F.one_hot(target_label, num_classes=num_labels).repeat(batch_size,
                                                                                       1)  # create one-hot target labels as (batch_size, num_class)

                # estimate the gradient
                grads, loss = get_grad_estimation(
                    model = model,
                    evaluate_img=adv_image,
                    target_labels=target_labels,
                    sample_per_draw=samples_per_draw,
                    batch_size=batch_size,
                    sigma = sigma,
                    targeted = targeted,
                    norm_theshold = norm_theshold,
                    host = host
                    )

                # # compute the true gradient
                # grads, loss = torch.func.grad_and_value(compute_loss)(adv_image, target_labels)

                last_ls.append(loss)
                last_ls = last_ls[-plateu_length:]
                if last_ls[-1] >= last_ls[0] and len(last_ls) == plateu_length:
                    samples_per_draw += batch_size
                    samples_per_draw = min(samples_per_draw, max_per_draw)
                    # print("alter the sample and learning rate.")
                    batch_size = samples_per_draw
                    k += round(k_hat * 0.9)
                    k = min(k, max_perturb)
                    k_hat *= 0.9
                    if max_learning_rate > min_learning_rate:
                        max_learning_rate = max(
                            max_learning_rate * 0.9, min_learning_rate)

                    # delta += torch.rand(channels,image_size,image_size).to(device_ids[0])
                    last_ls = []

                grads = admm_transformer.apply_gradient(grad=grads)
                # lr = next(scheduler)
                lr = max_learning_rate
                delta = delta - lr * grads
                # clip the delta to satisfy l_inf norm
                pro_delta = torch.clip(delta, lower_bond, uppder_bond)

                h = pro_delta ** 2 - 2 * pro_delta * \
                    delta  # shape: [1, image.shape]

                unclip_delta = greedy_project(h, delta, masks.clone(), k, d)
                unclip_delta = unclip_delta.resize(
                    channels, image_size, image_size)

                # # all pixels
                # # h = delta
                # pro_delta = pro_delta.flatten()
                # flatten_h = h.flatten()
                # min_k_idx = torch.topk(flatten_h, dim=0, k=d, largest=True).indices
                # delta_k = torch.zeros_like(pro_delta)
                # delta_k[min_k_idx] = pro_delta[min_k_idx]
                # delta = delta_k.reshape_as(target_img)
                # ################

                # clip the delta to satisfy l_inf norm
                delta = torch.clip(unclip_delta, lower_bond, uppder_bond)
                adv_image = torch.clip(target_img + delta, -0.5, 0.5)

                l0_norm = torch.sum((delta != 0).float())
                l2_norm = torch.norm(delta)
                linf_norm = torch.max(torch.abs(delta))
                num_queries += samples_per_draw+1

                last_query -= samples_per_draw+1
                if last_query - samples_per_draw-1 < 0:
                    break
                if iters+1 % cfg.log_iters == 0:
                    print('attack iter {}, loss: {:.5f}, group: {}， spd: {}, l0 norm:{:.5f}, l2 norm: {:.5f}, lr:{:.5f}, prediction: {}, target_label:{}'.format(
                        iters, loss.cpu(), k, samples_per_draw, l0_norm, l2_norm.cpu(), lr, pred.cpu(), target_label))
            else:
                # print("Fail Attack!")
                pass

            if targeted == 'untargeted':
                return adv_image, flag, num_queries, l0_norm.cpu(), l2_norm.cpu(), linf_norm.cpu(), pred.cpu()
            else:
                return adv_image, flag, num_queries, l0_norm.cpu(), l2_norm.cpu(), linf_norm.cpu(), target_label.cpu()

    def is_sucessful(target_label, pred):
        return (targeted == 'untargeted' and pred != target_label) or (targeted == 'targeted' and pred == target_label)

    num_queries_list = []
    l0_norm_list = []
    l2_norm_list = []
    linf_norm_list = []
    psnr_list = []
    ssim_list = []
    result = {}
    acc_count = 0

    if grouping_strategy == 'standard':
        # Standard grouping
        print("Standard Grouping!")
        if cfg.load_pre_groups:
            if if_overlap == "nonoverlapping":
                masks = torch.load(
                    f"Group/{dataset}/onehot_index_{dataset}_standard_{if_overlap}_{filtersize}.pth")
            else:
                masks = torch.load(
                    f"Group/{dataset}/onehot_index_{dataset}_standard_{if_overlap}_{filtersize}{stride}.pth")

            # index = torch.load(f"/home/yym/Documents/YYM/2/our/NES/Group/index_{dataset}_standard_{if_overlap}.pth")
        else:
            masks = standard_grouping(
                image_size, filtersize, stride, channels, d)
            if model_name == 'VT':
                torch.save(
                    masks, f"Group/{model_name}/onehot_index_{dataset}_standard_{if_overlap}.pth")
            else:
                torch.save(
                    masks, f"Group/{dataset}/onehot_index_{dataset}_standard_{if_overlap}.pth")
        masks = masks.reshape(-1, image_size, image_size,
                              channels).transpose(1, 3).flatten(1)
        print("Grouping Completion!")
        print("masks shape", masks.shape)

        # cv2.imwrite(os.path.join("test.png"), masks.cpu().numpy().reshape(1, image_size, image_size).transpose(1,2,0)*200)
    # elif grouping_strategy == "kmeans":
    #     G = 100
    #     print(f"This image is divided into {G} groups.")
    assert cfg.load_pre_groups == True

    i = 0
    index_fail = []
    for image, orig_label, target_label in zip(images, labels, target_class):
        # print("No. ",i+1)
        i += 1

        adv_image, flag, num_queries, l0_norm, l2_norm, linf_norm, target_label = attack(target_img=image.to(device_ids[0]),
                                                                                         target_label=target_label.to(
                                                                                             device_ids[0]),
                                                                                         original_label=orig_label.to(
                                                                                             device_ids[0]),
                                                                                         samples_per_draw=samples_per_draw,
                                                                                         batch_size=batch_size,
                                                                                         masks=masks.to(
                                                                                             device_ids[0]),
                                                                                         max_learning_rate=max_learning_rate,
                                                                                         cfg=cfg
                                                                                         )

        if flag:
            orig_img = img_transform(image.cpu().numpy())
            adv_img = img_transform(adv_image[0].cpu().numpy())
            psnr_list.append(calculate_psnr(orig_img, adv_img, dataset))
            ssim_list.append(calculate_ssim(orig_img, adv_img, dataset))
            l0_norm_list.append(l0_norm)
            l2_norm_list.append(l2_norm)
            linf_norm_list.append(linf_norm)
            acc_count += 1

            if cfg.save_image:
                image_save(adv_image[0], image, orig_label, target_label,
                           dataset + '_' + model_name, targeted, i, grouping_strategy)

        else:
            index_fail.append(i)

        num_queries_list.append(num_queries)

    acc = acc_count / torch.sum(correct_idx.float())
    print("fail_index:\n", index_fail)
    print("acc_count:", acc_count, "len(list)", torch.sum(correct_idx.float()))
    result = {key: value for key, value in [
        ('query', num_queries_list),
        ('acc', acc),
        ('l0', torch.mean(torch.tensor(l0_norm_list))),
        ('l2', torch.mean(torch.tensor(l2_norm_list))),
        ('linf', torch.mean(torch.tensor(linf_norm_list))),
        ('psnr', psnr_list),
        ('ssim', ssim_list)
    ]}

    return result
