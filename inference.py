import argparse
import os
import os.path as osp

import cv2
from imageio import imread, imsave
import numpy as np
import pytorch_lightning
import torch
from mmcv import Config
from torchvision.transforms import ToTensor
from tqdm import tqdm

from datasets import NUSCENES_ROOT
from models import MODELS
from models.utils import disp_to_depth
from utils import read_list_from_file, save_color_disp

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# output dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='weather')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--visualization', action='store_true')
    return parser.parse_args()

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225
    return array

if __name__ == '__main__':
    # parse args
    args = parse_args()
    # config
    cfg = Config.fromfile(osp.join('configs/', f'{args.config}.yaml'))
    # print message
    print('Now evaluating with {}...'.format(os.path.basename(args.config)))
    # device
    device = torch.device('cuda:0')
    # read list file
    test_items = read_list_from_file(osp.join(NUSCENES_ROOT['split'], '{}_test_split.txt'.format(args.root_dir)), 1)
    # store results
    predictions = []
    # model
    model_name = cfg.model.name
    net: pytorch_lightning.LightningModule = MODELS.build(name=model_name, option=cfg)
    net.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    net.to(device)
    net.eval()
    print('Successfully load weights from {}.'.format(args.checkpoint))
    # transform
    to_tensor = ToTensor()
    print(cfg.dataset['width'], cfg.dataset['height'])
    # no grad
    with torch.no_grad():
        # predict
        input_dir = '/data3/zyp/RNW/my_test/'
        img_list = os.listdir(input_dir)
        for i in tqdm(range(len(img_list))):
            # path
            # read image
            img_path = os.path.join("my_test/", img_list[i])
            rgb = cv2.imread(img_path)
            # resize
            rgb = cv2.resize(rgb, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
            # to tensor
            t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
            # feed into net
            outputs = net({('color_aug', 0, 0): t_rgb})
            disp = outputs[("disp", 0, 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            depth = depth.cpu()[0, 0, :, :].numpy()
            name = img_list[i].split(".")[0]
            save_path = os.path.join("/data3/zyp/RNW/save_day/numpy", name + ".npy")
            np.save(save_path, depth)
            # append
            # visualization
            vis = 1
            if vis == 1:
                scaled_disp = scaled_disp[0][0]
                rgb_disp_path = save_path = os.path.join("/data3/zyp/RNW/save_day/rgb_disp", name + ".png")
                rgb_depth_path = save_path = os.path.join("/data3/zyp/RNW/save_day/rgb_depth", name + ".png")
                disp = (255*tensor2array(scaled_disp, max_value=None, colormap='bone')).astype(np.uint8)
                imsave(rgb_disp_path, np.transpose(disp, (1, 2, 0)))

                depth = 1/scaled_disp
                depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                imsave(rgb_depth_path, np.transpose(depth, (1, 2, 0)))
            # if vis == 1:
            #     scaled_disp = scaled_disp.cpu()[0, 0, :, :].numpy()
            #     out_fn = os.path.join("/data3/zyp/RNW/save_day/rgb", '{}.png'.format(name))
            #     save_color_disp(rgb[:, :, ::-1], scaled_disp, out_fn, max_p=95, dpi=256)
    # stack
    # show message
    tqdm.write('Done.')
