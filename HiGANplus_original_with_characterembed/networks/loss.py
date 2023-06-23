import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import torchvision.transforms.functional as F_t
import cv2
import numpy as np

def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def tv_loss(img, img_lens):
    loss = (recn_l1_loss(img[:, :, 1:, :], img[:, :, :-1, :], img_lens) +
            recn_l1_loss(img[:, :, :, 1:], img[:, :, :, :-1], img_lens - 1)) / 2
    return loss


def recn_l1_loss(img1, img2, img_lens):
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * img_lens.sum())
    return loss

def recn_word_proj_loss(img1, img2, img_lens):
    #print('image: ', img1)
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    ### todo: resize images to 128x128
    img1 = img1 * mask.view(mask.size(0), 1, 1, mask.size(1))
    img2 = img2 * mask.view(mask.size(0), 1, 1, mask.size(1))
    ### todo: Plot images
    img2 = 255 - img2[1, :, :, :].squeeze(0)
    img2_loss = np.uint8(img2.detach().cpu().numpy())
    print('img_2loss:   ' , img2_loss)
    cv2.imwrite('img2_test.png', img2_loss)
    img1 = 255 - img1[1, :, :, :].squeeze(0)
    img1_loss = np.uint8(img1.detach().cpu().numpy())
    print('img_1loss:   ', img1_loss)
    cv2.imwrite('img1_test.png', img1_loss)
    exit(1)
    #print(img1.size())
    img1 = F_t.resize(img1, (128, 128))
    img2 = F_t.resize(img2, (128, 128))
    #print('resize image: ' , img1)
    angles = [0, 30, 60, 90, 120, 150]
    sum_kl = torch.zeros(size = (1,1), requires_grad=True).to('cuda')

    #bins = torch.torch.linspace(0, 255, 64)
    nbins = 64
    ### todo: rotate in 6 directions

    ### todo: rotate the image for angle [0, 30,  60, 90, 120, 150]
    for angle in angles:
        angle = torch.tensor(angle, dtype = torch.float32)
        angle = angle.repeat(img1.size()[0]).to('cuda')
        #print('angle: ', angle)
        img1_rotate = kornia.geometry.transform.rotate(img1, angle, center=None, mode='bilinear', padding_mode='zeros', align_corners=True)
        #F.rotate(img1,  angle)
        #print('img1_rotate', img1_rotate)
        img2_rotate = kornia.geometry.transform.rotate(img2, angle, center=None, mode='bilinear', padding_mode='zeros', align_corners=True)
        ### todo: histogram of rotated image
        #img1_rotate_histogram = torch.cat([torch.histc(x, bins=64, min=0, max=255) for x in img1_rotate], 0).view(8, 64)
        #img2_rotate_histogram = torch.cat([torch.histc(x, bins=64, min=0, max=255) for x in img2_rotate], 0).view(8, 64)
        img1_rotate_histogram = kornia.enhance.image_histogram2d(img1_rotate, n_bins= nbins, bandwidth=None, centers=None, return_pdf=False, kernel='triangular', eps=1e-10)[0]
        img2_rotate_histogram = kornia.enhance.image_histogram2d(img2_rotate, n_bins= nbins, bandwidth=None, centers=None, return_pdf=False, kernel='triangular', eps=1e-10)[0]
        #print('img1_rotate_histogram', img1_rotate_histogram)
        norm_phi_y = torch.norm(img1_rotate_histogram, dim=1) ## Generated
        norm_phi_y_hat = torch.norm(img2_rotate_histogram, dim=1) ### GT

        #print('norm phi y:   ', norm_phi_y)


        sum_kl +=torch.nn.functional.kl_div(norm_phi_y,norm_phi_y_hat, size_average=None, reduce=None, reduction='mean', log_target=False)\
                 #/(img1.size(1) * img1.size(2) * img_lens.sum())
        #    input, target (img2_rotate_histogram/norm_phi_y, img1_rotate_histogram/norm_phi_y_hat)
        #print('sum kl:   ', sum_kl)
    proj_char_loss = sum_kl/len(angles)
    print('sum proj char loss:   ',  proj_char_loss)

    return proj_char_loss

def calc_loss_perceptual(hout, hgt, img_lens):
    loss = 0
    for j in range(3):
        scale = 2 ** (3 - j)
        loss += recn_l1_loss(hout[j], hgt[j], img_lens // scale) / scale
    return loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def KLloss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


##############################################################################
# Contextual loss
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.5, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)
        CX = torch.mean(CX.max(dim=3)[0].max(dim=2)[0], dim=1)
        CX = torch.mean(-torch.log(CX + 1e-5))
        return CX



##############################################################################
# Gram style loss
##############################################################################
class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input_feat, target_feat, feat_len=None):
        input_gram = self.gram(input_feat, feat_len)
        target_gram = self.gram(target_feat, feat_len)
        loss = self.criterion(input_gram, target_gram)
        return loss


class GramMatrix(nn.Module):
    def forward(self, input, feat_len=None):
        a, b, c, d = input.size()

        if feat_len is not None:
            # mask for varying lengths
            mask = _len2mask(feat_len, d).view(a, 1, 1, d)
            input = input * mask

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        return G.div(a * b * c * d)
