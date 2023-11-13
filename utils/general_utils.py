import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from optimizer import cos_scheduler, step_lr_scheduler, loss_lr, clwars
from utils.select_groups import greedy_project

def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

def img_transform(img):
    """transfrom the img to cv2 format, with BGR and value clipping

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(img, np.ndarray), 'img type must a numpy array'
    # assert len(img.shape) == 3, 'img must be contain dim for channels'

    # move the channel to the last of dim
    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
    
    # transfrom the img from RGB2BGR
    # img = img[...,::-1]
    if img.dtype == np.float32:
        # img = img / np.max(img)
        img = (img + 0.5) *255
        img= img.astype(np.uint8)
    if img.dtype == np.int8:
        img = np.clip(img, 0, 255)
    # cv2.imwrite("cifar/qq.png",img*255)
    return img

def img_save(image, image_id, typ, group_strategy, dataset, orig_label, target_label):
    path = "RESULT/" + dataset+"/"+ group_strategy + "/" + typ + "/" 
    name = typ + "_index"+str(image_id)+ "_origin" + str(orig_label) + "_target" + str(target_label) + ".png"
    if not os.path.exists(path):
        os.makedirs(path)

    if 'mnist' in dataset:
        image = image.reshape(28,28).astype(np.int32)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.convert('L') # pat attention to its mode
        image_pil.save(path + name)
    else:
        # matplot
        image = image.astype(np.uint8)
        # plt.imshow(image)
        plt.imsave(path+name, image)

def calculate_psnr(original_img, reconstructed_img, dataset):
    # Ensure that the images have the same dimensions and data type
    if original_img.shape != reconstructed_img.shape:
        raise ValueError("Images should have the same dimensions.")
    if original_img.dtype != reconstructed_img.dtype:
        raise ValueError("Images should have the same data type.")
    
    # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
    # Calculate the MSE (Mean Squared Error) for each color channel
    if dataset != 'mnist':
        mse_r = np.mean((original_img[:, :, 0] - reconstructed_img[:, :, 0]) ** 2)
        mse_g = np.mean((original_img[:, :, 1] - reconstructed_img[:, :, 1]) ** 2)
        mse_b = np.mean((original_img[:, :, 2] - reconstructed_img[:, :, 2]) ** 2)
        
        # Calculate the average MSE across all channels
        mse_avg = (mse_r + mse_g + mse_b) / 3.0
        
        # Calculate the maximum possible pixel value based on the image data type
        max_pixel_value = np.iinfo(original_img.dtype).max
    
        # Calculate PSNR using the average MSE
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse_avg))
    else:
        mse = np.mean((original_img - reconstructed_img) ** 2)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
    return psnr

def calculate_ssim(img1, img2, dataset):
    # Ensure both images have the same data type and range
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    
    # Compute SSIM
    if dataset != 'mnist':
        ssim_index = ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min(), gaussian_weights=True, win_size=13,channel_axis=2)
    else:
        img1 = img1[0]
        img2 = img2[0]
        ssim_index = ssim(img1, img2)

    return ssim_index

def delta_initialization(delta, mask, k, d):
    h = - delta ** 2
    unclip_delta = greedy_project(h, delta, mask, k, d)
    return unclip_delta

lr_scheduler_dict={
    'coslr': cos_scheduler,
    'steplr': step_lr_scheduler,
    'losslr': loss_lr,
    'clwars': clwars
}

def get_lr_scheduler(**kwargs):
    name = kwargs['name']
    del kwargs['name']
    return lr_scheduler_dict[name](**kwargs)

