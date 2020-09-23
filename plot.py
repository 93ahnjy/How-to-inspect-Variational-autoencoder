import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from modules import *




def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x






def plot_result(images_n, images_r, num, name):

    for i in range(len(images_n)):
        input_img = to_img(images_n[i].cpu().permute(1, 2, 0).data)
        output_img = to_img(images_r[i].cpu().permute(1, 2, 0).data)

        plt.subplot(121)
        plt.imshow(input_img.squeeze(2))
        plt.subplot(122)
        plt.imshow(output_img.squeeze(2), vmax=1, vmin=0)
        plt.savefig(name, dpi=300)
        plt.clf()

        if i + 1 == num:
            return


def plot_result_array(images_n, images_r, num):

    images_n = torch.cat(images_n, dim=0)
    images_r = torch.cat(images_r, dim=0)

    #ssim_img = SSIM_map(images_n, images_r, window_size=11, sigma=0).cpu().mean(dim=1)
    images_n = images_n.cpu()
    images_r = images_r.cpu()


    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #window   = gaussian_kernel(5, sigma=1.5, color_channel=3).to(device)
    #images_n = F.conv2d(images_n, window, padding=5 // 2, groups=3).cpu()


    for i in range(len(images_n)):
        input_img = to_img(images_n[i].permute(1, 2, 0).data)
        output_img = to_img(images_r[i].permute(1, 2, 0).data)


        plt.subplot(221)
        plt.imshow(input_img)
        plt.subplot(222)
        plt.imshow(output_img)
        plt.subplot(223)


        residual = torch.abs(input_img - output_img).pow(2).mean(dim=2)
        # residual = torch.abs(input_img - output_img).max(dim=2)[0]


        plt.imshow(residual)
        plt.savefig('./result/fig_test_{}.png'.format(i), dpi=300)
        print("Save image - {}".format(str(i)))
        plt.clf()

        if i + 1 == num:
            return




def plot_result_with_attention(images_n, images_r, attention_map, num, name):

    for i in range(len(images_n)):
        input_img = to_img(images_n[i].cpu().permute(1, 2, 0).data)
        output_img = to_img(images_r[i].cpu().permute(1, 2, 0).data)

        att_img = attention_map[i].cpu().data.unsqueeze(0)
        att_img = F.upsample(att_img, scale_factor=16, mode='bilinear')

        vmax = torch.max(att_img)
        vmin = vmax / 2

        plt.subplot(121)
        plt.imshow(input_img.squeeze(2))
        plt.subplot(122)
        plt.imshow(output_img.squeeze(2))
        plt.imshow(att_img.squeeze(), alpha=0.2, cmap='jet', interpolation='bilinear', vmax=vmax, vmin=vmax * 0.5)
        plt.savefig(name, dpi=300)
        plt.clf()

        if i + 1 == num:
            return






def plot_result_with_attention_one_image(img_n, img_r, attention_map, name):


    input_img  = img_n.cpu().permute(1, 2, 0).data
    output_img = img_r.cpu().permute(1, 2, 0).data
    res = torch.abs((input_img  - output_img + 1e-4).max(dim=2)[0])


    att_img = attention_map.cpu().data.unsqueeze(0)


    c,h,w = img_n.size()
    att_img = F.upsample(att_img, size=(h,w), mode='bilinear',align_corners=True)


    vmax = torch.max(att_img)
    vmin = torch.min(att_img)
    print(vmax, vmin)



    plt.subplot(221)
    plt.imshow(input_img.squeeze(2))
    #plt.imshow(att_img.squeeze(), alpha=0.3, cmap='jet', interpolation='bilinear')

    plt.subplot(222)
    plt.imshow(output_img.squeeze(2))
    #plt.subplot(133)
    # plt.imshow(att_img.squeeze(), interpolation='bilinear')


    plt.subplot(223)
    plt.imshow(input_img.squeeze(2))
    plt.imshow(res, alpha=0.8, cmap='jet', interpolation='bilinear')
    #plt.imshow(res*att_img.squeeze() + res)


    plt.subplot(224)
    plt.imshow(input_img.squeeze(2), cmap='gray')
    plt.imshow(att_img.squeeze(), alpha=0.5, cmap='jet', vmax=vmax, vmin=vmin)
    #plt.imshow(res*att_img.squeeze() + res, alpha=0.8, cmap='jet')


    plt.savefig(name, dpi=300)
    plt.clf()





def plot_with_attention_grid(batch_attention, img_n, name):

    b,c,h,w = batch_attention.size()

    img_r   = to_img(img_n.cpu().unsqueeze(0).data)
    img_r   = img_r.repeat(b, 1, 1, 1)                   # [10, 3, 32, 32]
    att_img = batch_attention.cpu().data


    img_r   = torchvision.utils.make_grid(img_r, nrow=5, padding=1).permute(1,2,0)
    att_img = torchvision.utils.make_grid(att_img, nrow=5, padding=1, normalize=False, scale_each=False).permute(1,2,0).mean(2)


    vmax = torch.max(att_img)
    vmin = torch.min(att_img)



    plt.imshow(img_r)
    plt.imshow(att_img, alpha=0.3, cmap='jet', vmax=vmax, vmin=vmin)
    plt.savefig(name, dpi=300)
    plt.clf()