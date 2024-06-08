
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
from natsort import natsorted
from glob import glob
import utils

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import cv2
import argparse
from matplotlib import pyplot as plt
import time
from visualizer import get_local
get_local.activate() # 激活装饰器
from model.four_wav import LLFormer
parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/LOLv1/test/low', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/v1_inverse/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./checkpoints/Mamba_LOL_v1_inverse/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

"""
    visualization(feature map)
"""
def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def visulize_attention_ratio(img, attention_mask, filename, ratio=0.5, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask *255).astype('uint8')
    heatmap = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    superimposed_img1 = heatmap * 0.26 + img
    cv2.imwrite("./result_layer/"+filename+".jpg", superimposed_img1)
    print(normed_mask.shape)

def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size

    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def hook_func(module, input):
    x = input[0][0]
    x = x.unsqueeze(1)
    global i
    image_batch = torchvision.utils.make_grid(x, padding=4)
    image_batch = image_batch.cpu().numpy().transpose(1, 2, 0)
    writer.add_image("test", image_batch, i, dataformats='HWC')
    i += 1




def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(#glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
writer = SummaryWriter("./logs")
model = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,3,4],heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = False)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from torchsummary import summary
# summary(model.to(device), input_size=(3, 512, 512), batch_size=-1)

print('restoring images......')

mul = 16
index = 0
psnr_val_rgb = []
for file_ in files:
    img = Image.open(file_).convert('RGB')
    if img.width/1024 >= 3.5 or img.height/1024 >= 3.5:
        img = img.resize((int(img.width * 0.25), int(img.height * 0.25)), Image.ANTIALIAS)
    elif img.width/1024 >= 3 or img.height/1024 >= 3:
        img = img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.ANTIALIAS)
    elif img.width/1024 >= 1.1 or img.height/1024 >= 1.1:
        img = img.resize((int(img.width * 0.5), int(img.height * 0.5)), Image.ANTIALIAS)
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    print(input_.size())
    time_start = time.time()
    with torch.no_grad():
        restored = model(input_)
        # cache = get_local.cache
        # # atten_mask_maps = cache['TransformerBlock_WithMask.forward']
        # # atten_out_maps = cache['TransformerBlock_WithoutMask.forward']
        # # visulize_attention_ratio(img, attention_mask=atten_mask_maps[5][0, 0, 0:, 0:],filename="mask")
        # # visulize_attention_ratio(img, attention_mask=atten_out_maps[25][0, 2, 0:, 0:],filename="out")
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '_2.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
