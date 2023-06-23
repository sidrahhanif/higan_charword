import torch
import kornia as K
import torchvision.transforms as transforms
import numpy as np
import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        input_tensor = torch.tensor(input)
        threshold_tensor = torch.tensor(threshold)
        ctx.save_for_backward(input_tensor, threshold_tensor)
        output = torch.where(input_tensor < threshold_tensor, torch.ones_like(input_tensor), torch.zeros_like(input_tensor))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, threshold_tensor = ctx.saved_tensors
        grad_input = grad_threshold = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        if ctx.needs_input_grad[1]:
            grad_threshold = -(input_tensor < threshold_tensor).float() * grad_output
        return grad_input, grad_threshold

class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        return BinarizeFunction.apply(input, self.threshold)

def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def kl_1d(a, b):  # [N, C]
    eps = 1e-8
    return torch.nn.functional.kl_div(torch.log(a + eps), b, reduction='mean')

def w_dis_1d(a, b): # [N, C]
    a_cdf = torch.cumsum(a, dim=1) # [N, C]
    a_cdf = a_cdf / a_cdf[:,-1:]
    b_cdf = torch.cumsum(b, dim=1)
    b_cdf = b_cdf / b_cdf[:,-1:]
    return (a_cdf - b_cdf).abs().sum(1) # [N]

def visualize_hist(f1r_1, f2r_1, l , pre):
    # Assume data is the list or array of values you want to plot the histogram of
    data = f1r_1
    labels = range(len(data))
    # Compute the histogram counts
    #hist, bins = np.histogram(data, bins=nbins)

    # Plot the histogram
    #plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge')
    plt.bar(labels, data)
    # Add labels and titles
    plt.title("Histogram of Data")
    plt.xlabel("Bins")
    plt.ylabel("Counts")

    # Save the plot to a file
    plt.savefig(pre + "_histogram_rot_" + l + '_0_' +"_.png")

    # Show the plot
    plt.show()
    ## histogram 2
    data = f2r_1
    bins = range(len(data))
    # Compute the histogram counts
    #hist, bins = np.histogram(data, bins=nbins)
    #print(len(bins), len(data))
    # Plot the histogram
    #plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge')
    plt.bar(labels, data)
    # Add labels and titles
    plt.title("Histogram of Data")
    plt.xlabel("Bins")
    plt.ylabel("Counts")

    # Save the plot to a file
    plt.savefig(pre + "_histogram_rot_" + l + '_1_' +"_.png")

    # Show the plot
    plt.show()

def transform_crop_resize(image, boxes, size):
    cropped_image = K.geometry.transform.crop_and_resize(image, boxes, (size, size),
                                                         padding_mode='border')
    #print('cropped image size', cropped_image.size())
    return cropped_image

def convert_imglens_into_bboxes(img_lens):
    ### todo: convert image len into bboxes
    complete_list = []
    for l in img_lens:
        T = torch.tensor([[0, 0], [l, 0], [l,64], [0, 64]])
        complete_list.append(T)
        #print('complete_tensor_size:', complete_list)
    complete_tensor = torch.stack(complete_list, dim=0)
    return complete_tensor
def PKL(f1, f2, img_lens, threshold):  # N,1,H,W
    # f1, f2 in [-1, 1] fg -1 bg 1
    # W distance Loss
    # HW
    ### todo: f1 => recn image, f2 => real image
    #f1 = (1 - f1) / 2  # fg 1 bg 0
    #f2 = (1 - f2) / 2  # fg 1 bg 0
    binarize = Binarize(threshold=0.7)
    mean, std = 0.5, 0.5
    i = 3
    epsilon = 0
    #mask = _len2mask(img_lens, f1.size(-1)).to(f1.device)

    #f1 = f1 * mask.view(mask.size(0), 1, 1, mask.size(1))
    #f2 = f2 * mask.view(mask.size(0), 1, 1, mask.size(1))
    ### todo: crop f2 and f1 image by len
    #f1  = K.geometry.crop.crop_by_box(f1, bbox)
    #cv2.imwrite('f2.png', np.asarray(f2[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #cv2.imwrite('f1.png', np.asarray(f1[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #exit(1)
    #print(f)
    ### todo: unnormalize GT images
    unnormalize = transforms.Normalize((-1 * mean) / std, 1.0 / std)
    transform = transforms.Compose([unnormalize])
    #print('img lens:',img_lens)
    f1, f2 = transform(f1), transform(f2)
    boxes = convert_imglens_into_bboxes(img_lens)
    #print('boxes size:', boxes.size())
    size = 128
    f1 = transform_crop_resize(f1, boxes, size)
    f2 = transform_crop_resize(f2, boxes, size)
    #cv2.imwrite('f2_crop_resize.png', np.asarray(f2[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #cv2.imwrite('f1_crop_resize.png', np.asarray(f1[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    # cv2.imwrite('f1_mask.png', np.asarray(f1[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #exit(1)
    # Apply the transform to the image tensor
    f2_inv = torch.neg(f2)
    f2_inv = (f2_inv + 1) / 2
    f1_inv = torch.neg(f1)
    f1_inv = (f1_inv + 1) / 2
    #print('f1 inv:', f1_inv)
    #cv2.imwrite('f2_inv.png', np.asarray(f2_inv[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #cv2.imwrite('f1_inv.png', np.asarray(f1_inv[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    ### todo: resize f1 AND f2 TO FIXED SIZE of 128x128
    #f1 = F.resize(f1_inv, (80,80))  ### nbins = 128
    #f2 = F.resize(f2_inv, (80,80))
    #print('f1 resize:', f1)
    #cv2.imwrite('f2_inv.png', np.asarray(f2_inv[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #cv2.imwrite('f1_inv.png', np.asarray(f1_inv[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #exit(1)
    #######################################################################
    ### todo: binary image and resize
    f2_max_value = torch.max(f2_inv)
    f2_normalized_tensor = (f2_inv.float() / f2_max_value.float())
    f1_max_value = torch.max(f1_inv)
    f1_normalized_tensor = (f1_inv.float() / f1_max_value.float())
    #cv2.imwrite('f1_inv_norm.png', np.asarray(f1_normalized_tensor[i, :, :, :].squeeze(0).detach().cpu())*255)
    #cv2.imwrite('f2_inv_norm.png', np.asarray(f2_normalized_tensor[i, :, :, :].squeeze(0).detach().cpu())*255)
    #print('f1 normalize:', f1_normalized_tensor)
    #f1 = torch.where(f1_normalized_tensor > threshold)
    f1 = binarize(f1_normalized_tensor)
    f2 = binarize(f2_normalized_tensor)
    #print('f1 binary:', f1)
    #print('f2 binary:', f2)
    #f1, f2 = f1_normalized_tensor < threshold, f2_normalized_tensor < threshold
    #f1, f2 = f1.float(), f2.float()
    #cv2.imwrite('f2_inv_norm_th.png', np.asarray(f2[i, :, :, :].squeeze(0).detach().cpu()) * 255)
    #cv2.imwrite('f1_inv_norm_th.png', np.asarray(f1[i, :, :, :].squeeze(0).detach().cpu())*255)
    #cv2.imwrite('f2_inv_res_norm_th.png', np.asarray(f2[i, :, :, :].squeeze(0).detach().cpu())*255)
    #cv2.imwrite('f1_inv_res_norm_th.png', np.asarray(f1[i, :, :, :].squeeze(0).detach().cpu())*255)

    ######################################################################
    #print('img2 size:', f2_unnormalize.size())
    #cv2.imwrite('f1_unnorm.png', np.asarray(f2_unnormalize.squeeze(0).detach().cpu()) * 255)
    #exit(1)
    ### todo: histogram code
    B = f1.shape[0]
    #print(f1.size())
    f1_0 = f1.sum((1, 2)) + epsilon  # N,W
    f2_0 = f2.sum((1, 2)) + epsilon  # N,W
    loss_0 = kl_1d(f2_0, f1_0)  # N
    #print('loss_0', loss_0)
    f1_1 = f1.sum((1, 3)) + epsilon  # N,H
    f2_1 = f2.sum((1, 3)) + epsilon  # N,H
    #visualize_hist(f1_0[i].detach().cpu().numpy(), f1_1[i].detach().cpu().numpy(), '1', 'o')
    #visualize_hist(f2_0[i].detach().cpu().numpy(), f2_1[i].detach().cpu().numpy(), '2', 'o')
    loss_1 = kl_1d(f2_1, f1_1)  # N
    losses = [loss_0, loss_1]
    #print('loss_1', loss_1)
    #exit(1)
    for angle in [15., 30., 45., 60., 75.]:
        angle = angle * torch.ones(B, device=f1.device)
        f1r = K.geometry.rotate(f1, angle)
        f2r = K.geometry.rotate(f2, angle)
        #cv2.imwrite('f2_inv_norm_th_rot' +str(angle) + '.png', np.asarray(f2r[i, :, :, :].squeeze(0).detach().cpu()) * 255)
        #cv2.imwrite('f1_inv_norm_th_rot' +str(angle) + '.png', np.asarray(f1r[i, :, :, :].squeeze(0).detach().cpu()) * 255)
        #exit(1)
        f1r_0 = f1r.sum((1, 2))  # N,W
        f2r_0 = f2r.sum((1, 2))  # N,W
        lossr_0 = kl_1d(f2r_0, f1r_0)  # N
        losses.append(lossr_0)
        f1r_1 = f1r.sum((1, 3))  # N,H
        f2r_1 = f2r.sum((1, 3))  # N,H
        ### todo: histogram visualization
        #visualize_hist(f1r_0[i].detach().cpu().numpy(), f1r_1[i].detach().cpu().numpy(), '1', str(angle))
        #visualize_hist(f2r_0[i].detach().cpu().numpy(), f2r_1[i].de0tach().cpu().numpy(), '2', str(angle))
        #exit(1)
        lossr_1 = kl_1d(f2r_1, f1r_1)
        #print('lossr_0', lossr_0)
        #print('lossr_1', lossr_1)# N
        losses.append(lossr_1)
    loss = torch.stack(losses).mean()/size

    #print('loss:', loss)
    #exit(1)
    return loss

def calc_pkl(predict, target):
    return torch.mean(PKL(predict.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)))