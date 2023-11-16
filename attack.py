import torch
import numpy as np
from omegaconf import DictConfig
from optimizer import Momentum, margin_loss, Adam, cos_scheduler_increase
from torch.nn import functional as F
from utils.general_utils import *
from torch.nn import DataParallel
import os
import shutil
from utils.get_grad import ZOEstimator
from utils.select_groups import *
import warnings
warnings.filterwarnings('ignore')

loss_fn = margin_loss
SEED = 0
torch.random.initial_seed()
torch.random.manual_seed(SEED)
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


def init_delta_fun(name, shape, device, lower_bond, uppder_bond, epsilon, masks, k, d, last_delta, target_img):
    if name == 'default':
        # delta for initial
        delta = torch.rand(shape).to(device)
        adv_image = target_img
        return delta, adv_image
    elif name == 'boundary':
        # boundary -epsilon, epsilon
        delta_sign = torch.bernoulli(
            0.5 * torch.ones(shape)).to(device)
        delta = (2*delta_sign-1)*epsilon
        delta = delta_initialization(delta, masks.clone(), k, d)
        delta = delta.resize(*shape) + torch.rand(shape).to(device)*0.1
        delta = torch.clip(delta, lower_bond, uppder_bond)
    elif name == 'random':
        # random noist for ininital with k groups
        delta = torch.rand(shape).to(device)
        delta = delta_initialization(delta, masks.clone(), k, d)
        delta = delta.resize(*shape)
        delta = torch.clip(delta, lower_bond, uppder_bond)
    elif name == 'last_delta':
        # last delta for initial
        delta = last_delta if last_delta is not 0 else torch.rand(
            shape).to(device)
        delta = torch.clip(delta, lower_bond, uppder_bond)*0.5
    adv_image = torch.clip(target_img + delta, -0.5, 0.5)
    return delta, adv_image


def esal(
        images: np.ndarray,
        labels: np.ndarray,
        model: torch.nn.Module,
        cfg: DictConfig,):
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
    last_delta = 0

    # algorithm
    algorithm = cfg.setup.algorithm
    sigma = algorithm.sigma  # refers to sigma in the paper
    sample_num = algorithm.samples_per_draw
    # max batch size to evaluate the output
    max_sample_num_per_forward = algorithm.batch_delta_size
    plateu_length = algorithm.plateu_length
    grouping_strategy = algorithm.grouping_strategy
    if_overlap = algorithm.if_overlap
    epsilon = algorithm.epsilon  # correspond to epsilon in the paper
    perturb_pixels_ratio = algorithm.perturb_rate
    k_init = algorithm.k

    # optimizer
    scheduler = cfg.scheduler
    max_learning_rate = scheduler.max_lr
    min_learning_rate = scheduler.min_lr

    momentum = cfg.setup.optimizer.momentum

    # dataset and model
    d_m = cfg.dataset_and_model
    channels = d_m.num_channels
    dataset = d_m.name
    model_name = d_m.dataset_and_model.model_type
    filtersize = torch.tensor(d_m.filterSize)
    stride = d_m.stride
    image_size = torch.tensor(d_m.image_size)  # image size
    num_classes = d_m.num_classes

    dims = channels*image_size*image_size
    shape = (channels, image_size, image_size)

    if dataset == 'imagenet':
        max_sample_num = 100
        if model_name == 'inceptionv3':
            image_size = torch.tensor(299)
        elif model_name == 'VIT':
            image_size = torch.tensor(224)
    else:
        max_sample_num = 64
    max_group_num = round(1.0*dims.item()/filtersize.item() /
                          filtersize.item()/channels*perturb_pixels_ratio)

    device = torch.device(
        f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    images = torch.tensor(images).float().to(device)
    labels = torch.tensor(labels).to(device)

    if targeted == 'targeted':
        print(
            f"It performs targeted attack on {dataset} dataset, {model_name} model!")
        target_class = torch.tensor(
            [pseudorandom_target(index, num_classes, orig_label)
             for index, orig_label in enumerate(labels)], dtype=torch.int64
        ).to(device)
    else:
        print(
            f"It performs untargeted attack on {dataset} dataset, {model_name} model!")
        target_class = labels

    # validation on model
    with torch.no_grad():
        score = model(images)
        pred = torch.argmax(score, dim=-1)
        torch.cuda.empty_cache()
        # remove_id = torch.where(pred != labels)
        # remove_error(remove_id)
        # print(pred)
        # print(labels)
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
            target_image: torch.Tensor,
            target_label: torch.Tensor,
            masks: torch.Tensor,
            cfg: DictConfig,
            last_delta: torch.Tensor):
        """perform easl attack on target img 

        Args:
            target_image (torch.Tensor): target attack img
            target_label (torch.Tensor): label for target attack
            masks (torch.Tensor): the mask for grouping
            max_learning_rate (float): max learning rate
            cfg (DictConfig): setup for algorithm
            last_delta (torch.Tensor): the delta for last attacking
        """
        with torch.no_grad():
            # initialization
            target_image = target_image.reshape((1,) + target_image.shape)
            gradient_transformer = Momentum(
                variabels=target_image, momentum=momentum)
            admm_transformer = Adam(target_image)
            lr_scheduler = get_lr_scheduler(**cfg.scheduler)
            alpha_scheduler = cos_scheduler_increase(cfg.alpha, 0.01, 1000, 0)
            num_queries = 0
            flag = False
            lower_bond = torch.maximum(
                torch.tensor(-epsilon).to(device), -0.5 - target_image)
            uppder_bond = torch.minimum(torch.tensor(
                epsilon).to(device), 0.5 - target_image)
            # target_labels = F.one_hot(target_label, num_classes=num_labels).repeat(batch_size,
            #                                                                        1)  # create one-hot target labels as (batch_size, num_class)

            remain_query = cfg.max_query
            delta, adv_image = init_delta_fun(cfg.initial_delta, shape, device,
                                              lower_bond, uppder_bond, epsilon, masks, sample_stragety.k, dims, last_delta, target_image)
            lr = cfg.scheduler.max_lr
            # # for iter in range(cfg.max_iters):
            for iter in range(cfg.max_query):

                # check if we can make an early stopping
                pred = torch.argmax(model(adv_image, ))
                num_queries += 1
                if is_sucessful(target_label, pred):
                    flag = True
                    print(
                        f'[succ] Iter: {iter}, groups: {sample_stragety.k}, query: {num_queries}, prediction: {pred}, target_label:{target_label}')
                    break

                # create one-hot target labels as (batch_size, num_class)
                target_labels = F.one_hot(target_label,
                                          num_classes=num_classes).repeat(grad_estimator.max_sample_num_per_forward, 1)

                grads, loss = grad_estimator.zo_estimation(
                    evaluate_img=adv_image,
                    target_labels=target_labels,
                    subspace_estimation=cfg.using_subspace,
                )
                grads = admm_transformer.apply_gradient(grad=grads)
                sample_num, k = sample_stragety.update_sample_strategy(loss)

                if scheduler.name == 'clwars':
                    lr = get_clwars_lr(
                        delta=delta, grads=grads, max_lr=lr, eta=cfg.dataset_and_model.eta)
                elif scheduler.name == 'losslr':
                    lr = loss_lr_schedulr.get_next_lr(loss)
                else:
                    lr = next(lr_scheduler)

                lr *= 2 if loss > 1 else 0.1

                delta = delta - lr * grads

                # get sparse k groups delta
                delta = get_k_gorups_delta(
                    delta=delta,
                    masks=masks,
                    lower_bound=lower_bond,
                    uppder_bound=uppder_bond,
                    k=k
                )

                adv_image = torch.clip(target_image + delta, -0.5, 0.5)

                num_queries += sample_stragety.sample_num
                remain_query -= sample_stragety.sample_num+1
                if remain_query <= sample_stragety.sample_num-1 and remain_query > 1:
                    sample_stragety.sample_num = remain_query if remain_query % 2 == 0 else remain_query-1
                    sample_stragety.max_sample_num_per_forward = sample_stragety.sample_num
                elif remain_query <= 0:
                    break
                grad_estimator.max_sample_num_per_forward = sample_num
                grad_estimator.sample_num = sample_num
                grad_estimator.previous_grads = grads.detach().clone()
                grad_estimator.sample_num = sample_stragety.sample_num
                grad_estimator.max_sample_num_per_forward = sample_stragety.sample_num
                grad_estimator.previous_grad_queue.append(grads.flatten())
                grad_estimator.alpha = next(alpha_scheduler)

                if iter % cfg.log_iters == 0:
                    print(
                        f'attack iter {iter}, loss: {loss.cpu():.5f}, group: {k}, spd: {sample_stragety.sample_num}, lr:{lr:.3f}')

            l0_norm = torch.sum((delta != 0).float())
            l2_norm = torch.norm(delta)
            linf_norm = torch.max(torch.abs(delta))
            if targeted == 'untargeted':
                return adv_image, flag, num_queries, l0_norm.cpu(), l2_norm.cpu(), linf_norm.cpu(), pred.cpu(), delta
            else:
                return adv_image, flag, num_queries, l0_norm.cpu(), l2_norm.cpu(), linf_norm.cpu(), target_label.cpu(), delta

    def is_sucessful(target_label, pred):
        return (targeted == 'untargeted' and pred != target_label) or (targeted == 'targeted' and pred == target_label)

    def get_k_gorups_delta(delta, masks, lower_bound, uppder_bound, k):
        # clip the delta to satisfy l_inf norm
        pro_delta = torch.clip(delta, lower_bound, uppder_bound)
        h = pro_delta ** 2 - 2 * pro_delta * delta  # shape: [1, image.shape]
        unclip_delta = greedy_project(h, delta, masks.clone(), k)
        unclip_delta = unclip_delta.resize(*shape)

        if cfg.attack_all_pixels:


            pro_delta = pro_delta.flatten()
            flatten_h = h.flatten()
        min_k_idx = torch.topk(flatten_h, dim=0, k=dims, largest=True).indices
        # delta_k = torch.zeros_like(pro_delta)
        # delta_k[min_k_idx] = pro_delta[min_k_idx]
        # delta = delta_k.reshape_as(target_img)
        # ################

        # clip the delta to satisfy l_inf norm
        delta = torch.clip(unclip_delta, lower_bound, uppder_bound)
        return delta

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
                image_size, filtersize, stride, channels, dims)
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
        print("No. ", i+1)
        i += 1

        # # initialize the zeoestimator
        grad_estimator = ZOEstimator(
            model=model,
            sample_num=sample_num,
            max_sample_num_per_forward=max_sample_num_per_forward,
            sigma=sigma,
            targeted=targeted,
            grad_clip_threshold=d_m.grad_norm_threshold,
            alpha=cfg.alpha,
            subspace_dim=cfg.subspace_dim,
            device=device
        )
        
        sample_stragety = SampleStragegyScheduler(
            sample_num=sample_num,
            plateu_length=plateu_length,
            max_sample_num=max_sample_num,
            k=k_init,
            k_increase=k_init,
            k_max=max_group_num,
        )

        loss_lr_schedulr = LossLRScheduler(
            max_lr=max_learning_rate,
            min_lr=min_learning_rate,
            plateu_length=plateu_length
        )

        adv_image, flag, num_queries, l0_norm, l2_norm, linf_norm, target_label, delta = attack(
            target_image=image.to(device),
            target_label=target_label.to(device),
            masks=masks.to(device),
            cfg=cfg,
            last_delta=last_delta
        )

        orig_img = img_transform(image.cpu().numpy())
        adv_img = img_transform(adv_image[0].cpu().numpy())
        
        if flag:
            acc_count += 1
            if cfg.save_image:
                image_save(adv_image[0], image, orig_label, target_label,
                           dataset + '_' + model_name, targeted, i, grouping_strategy)
            last_delta = delta
        else:
            index_fail.append(i)

        num_queries_list.append(num_queries)
        l0_norm_list.append(l0_norm)
        l2_norm_list.append(l2_norm)
        linf_norm_list.append(linf_norm)
        psnr_list.append(calculate_psnr(orig_img, adv_img, dataset))
        ssim_list.append(calculate_ssim(orig_img, adv_img, dataset))

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
