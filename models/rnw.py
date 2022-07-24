import os.path as osp

import pytorch_lightning
import torch.nn.functional as F
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np

from components import freeze_model, unfreeze_model, ImagePool, get_smooth_loss
from utils import EWMA
from .disp_net import DispNet
from .gan import GANLoss, NLayerDiscriminator
from .layers import SSIM, Backproject, Project
from .registry import MODELS
from .utils import *
from SCI.model import *
from transforms import EqualizeHist

def build_disp_net(option, check_point_path):
    # create model
    model: pytorch_lightning.LightningModule = MODELS.build(name=option.model.name, option=option)
    model.load_state_dict(torch.load(check_point_path, map_location='cpu')['state_dict'])
    model.freeze()
    model.eval()

    # return
    return model


@MODELS.register_module(name='rnw')
class RNWModel(LightningModule):
    """
    The training process
    """
    def __init__(self, opt):

        super(RNWModel, self).__init__()
        self.test = opt.test
        self.opt = opt.model
        self._equ_limit = 0.004
        self._to_tensor = ToTensor()

        # components
        self.gan_loss = GANLoss('lsgan')
        self.image_pool = ImagePool(50)
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.ego_diff = EWMA(momentum=0.98)

        # networks
        self.S = Network(stage=3)
        self.G = DispNet(self.opt)
        in_chs_D = 3 if self.opt.use_position_map else 1
        self.D = NLayerDiscriminator(in_chs_D, n_layers=3)

        # init SCI_Model
        self.S.enhance.in_conv.apply(self.S.weights_init)
        self.S.enhance.conv.apply(self.S.weights_init)
        self.S.enhance.out_conv.apply(self.S.weights_init)
        self.S.calibrate.in_conv.apply(self.S.weights_init)
        self.S.calibrate.convs.apply(self.S.weights_init)
        self.S.calibrate.out_conv.apply(self.S.weights_init)

        # register image coordinates
        if self.opt.use_position_map:
            h, w = self.opt.height, self.opt.width
            height_map = torch.arange(h).view(1, 1, h, 1).repeat(1, 1, 1, w) / (h - 1)
            width_map = torch.arange(w).view(1, 1, 1, w).repeat(1, 1, h, 1) / (w - 1)

            self.register_buffer('height_map', height_map, persistent=False)
            self.register_buffer('width_map', width_map, persistent=False)

        # build day disp net
        self.day_dispnet = build_disp_net(
            Config.fromfile(osp.join('configs/', f'{self.opt.day_config}.yaml')),
            self.opt.day_check_point
        )

        # link to dataset
        self.data_link = opt.data_link

        # manual optimization
        self.automatic_optimization = False

    def forward(self, inputs):
        if not self.test:
            return self.G(inputs)
        else:
            self.S.eval()
            sci_gray = inputs[("color_gray", 0, 0)][0].unsqueeze(0)
            sci_color = inputs[("color", 0, 0)][0].unsqueeze(0)                    
            illu_list, _, _, _ = self.S(sci_gray)
            illu = illu_list[0][0][0]
            illu = torch.stack([illu, illu, illu])
            illu = illu.unsqueeze(0)
            r = sci_color / illu
            r = torch.clamp(r, 0, 1)
            inputs[("color_aug", 0, 0)] = r
            return self.G(inputs)



    def generate_gan_outputs(self, day_inputs, outputs):
        # (n, 1, h, w)
        night_disp = outputs['disp', 0, 0]
        with torch.no_grad():
            day_disp = self.day_dispnet(day_inputs)['disp', 0, 0]
        # remove scale
        night_disp = night_disp / night_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        day_disp = day_disp / day_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # image coordinates
        if self.opt.use_position_map:
            n = night_disp.shape[0]
            height_map = self.height_map.repeat(n, 1, 1, 1)
            width_map = self.width_map.repeat(n, 1, 1, 1)
        else:
            height_map = None
            width_map = None
        # return
        return day_disp, night_disp, height_map, width_map

    def compute_G_loss(self, night_disp, height_map, width_map):
        G_loss = 0.0
        #
        # Compute G loss
        #
        freeze_model(self.D)
        if self.opt.use_position_map:
            fake_day = torch.cat([height_map, width_map, night_disp], dim=1)
        else:
            fake_day = night_disp
        G_loss += self.gan_loss(self.D(fake_day), True)

        return G_loss

    def compute_D_loss(self, day_disp, night_disp, height_map, width_map):
        D_loss = 0.0
        #
        # Compute D loss
        #
        unfreeze_model(self.D)
        if self.opt.use_position_map:
            real_day = torch.cat([height_map, width_map, day_disp], dim=1)
            fake_day = torch.cat([height_map, width_map, night_disp.detach()], dim=1)
        else:
            real_day = day_disp
            fake_day = night_disp.detach()
        # query
        fake_day = self.image_pool.query(fake_day)
        # compute loss
        D_loss += self.gan_loss(self.D(real_day), True)
        D_loss += self.gan_loss(self.D(fake_day), False)

        return D_loss

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim_G, optim_D = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # get input data
        day_inputs = batch_data['day']
        night_inputs = batch_data['night']
        
        # TODO: get relight img
        night_inputs, sci_loss_dict = self.get_sci_relight(night_inputs)
        
        # night_inputs = self.get_mcie_relight(night_inputs)
        
        # # outputs of G
        # for key in night_inputs.keys():
        #     night_inputs[key] = night_inputs[key].detach()
        outputs = self.G(night_inputs)

        # loss for ego-motion
        disp_loss_dict = self.compute_disp_losses(night_inputs, outputs)
        
        # generate outputs for gan
        day_disp, night_disp, height_map, width_map = self.generate_gan_outputs(day_inputs, outputs)
        
        #
        # optimize G
        #
        # compute loss
        G_loss = self.compute_G_loss(night_disp, height_map, width_map)        
        S_loss = sum(sci_loss_dict.values())
        disp_loss = sum(disp_loss_dict.values())

        # log
        logger.add_scalar('train/disp_loss', disp_loss, self.global_step)
        logger.add_scalar('train/G_loss', G_loss, self.global_step)
        logger.add_scalar('train/S_loss', S_loss, self.global_step)

        # optimize G
        G_loss = G_loss * self.opt.G_weight + disp_loss + S_loss * 5
        # G_loss = S_loss * 0.05
        
        optim_G.zero_grad()
        self.manual_backward(G_loss)
        optim_G.step()

        #
        # optimize D
        #
        # compute loss
        D_loss = self.compute_D_loss(day_disp, night_disp, height_map, width_map)

        # # log
        logger.add_scalar('train/D_loss', D_loss, self.global_step)

        D_loss = D_loss * self.opt.D_weight

        # optimize D
        optim_D.zero_grad()
        self.manual_backward(D_loss)
        optim_D.step()

        # return G_loss + D_loss

    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D = self.lr_schedulers()

        sch_G.step()
        sch_D.step()

        self.data_link.when_epoch_over()

    def configure_optimizers(self):
        optim_params = [
            {'params': self.G.parameters(), 'lr': self.opt.learning_rate},
            {'params': self.S.parameters(), 'lr': 0.0003, 'betas': (0.9, 0.999), 'weight_decay': 3e-4}
        ]
        optim_G = Adam(optim_params)
        optim_D = Adam(self.D.parameters(), lr=self.opt.learning_rate)
        # optim_S = Adam(self.S.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=3e-4)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)

        return [optim_G, optim_D], [sch_G, sch_D]
    
    def get_sci_relight(self, inputs):

        loss_dict = {}
        src_colors = inputs[('color', 0, 0)]
        b, _, h, w = src_colors.shape
        
        # get sci_color 
        for scale in self.opt.scales:

            for frame_id in self.opt.frame_ids:
                sci_gray = inputs[("color_gray", frame_id, scale)]
                sci_color = inputs[("color", frame_id, scale)]
                loss, illu_list = self.S._loss(sci_gray, frame_id) #todo: index 0 loss, index 0, -1, 1 img
                if frame_id == 0:
                    loss_dict[("sci_loss", 0, scale)] = loss / len(self.opt.scales)
                
                illu = illu_list[0][0][0]
                
                if scale == 0 and frame_id == 0:
                    mask_high = illu <= 0.9
                    mask_low = illu >= 0.35
                    light_mask = mask_low * mask_high
                    inputs[("light_mask", frame_id, scale)] = light_mask

                illu = torch.stack([illu, illu, illu])
                illu = illu.unsqueeze(0)
                r = sci_color / illu
                r = torch.clamp(r, 0, 1)
                inputs[("color_aug", frame_id, scale)] = r
                inputs[("color_equ", frame_id, scale)] = r

                
        
        return inputs, loss_dict
        
        # get clahe enhancement for sci_color
    
    def get_mcie_relight(self, inputs):
        
        src_colors = inputs[('color_aug', 0, 0)]
        b, _, h, w = src_colors.shape
        src_colors = src_colors.cpu().detach().float().numpy()

        for scale in self.opt.scales:
            rh, rw = h // (2 ** scale), w // (2 ** scale)
            inputs_equ = {}
            for frame_id in self.opt.frame_ids:
                inputs_equ[frame_id] = []
            
            for batch_idx in range(b):
                src_color = src_colors[batch_idx]
                src_color = np.clip(src_color * 255.0, 0, 255.0).astype('uint8')
                src_color = src_color.transpose(1, 2, 0)
                equ_hist = EqualizeHist(src_color, limit=self._equ_limit)              
                
                for frame_id in self.opt.frame_ids:
                    sci_color = inputs[("color_aug", frame_id, scale)][batch_idx]
                    sci_color = sci_color.cpu().float().numpy() 
                    sci_color = np.clip(sci_color * 255.0, 0, 255.0).astype('uint8')
                    sci_color = sci_color.transpose(1, 2, 0)
                    equ_color = equ_hist(sci_color)
                    equ_color = self._to_tensor(equ_color)
                    if scale != 0:
                        equ_color = F.interpolate(equ_color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)

                    inputs_equ[frame_id].append(equ_color)
                
            for frame_id in self.opt.frame_ids:
                sci_equ = torch.cat(inputs_equ[frame_id]).cuda()
                if len(sci_equ.shape )== 3:
                    sci_equ = sci_equ.unsqueeze(0)
                inputs[('color_equ', frame_id, scale)] = sci_equ

        return inputs

    
    def get_color_input(self, inputs, frame_id, scale):
        return inputs[("color_equ", frame_id, scale)] if self.opt.use_equ else inputs[("color", frame_id, scale)]

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs["inv_K", 0])
            pix_coords = self.project_3d(cam_points, inputs["K", 0], T)  # [b,h,w,2]
            src_img = self.get_color_input(inputs, frame_id, 0)
            outputs[("color", frame_id, scale)] = F.grid_sample(src_img, pix_coords, padding_mode="border",
                                                                align_corners=False)
        return outputs

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_static_mask(self, pred, target):
        # compute threshold
        mask_threshold = self.ego_diff.running_val
        # compute diff
        diff = (pred - target).abs().mean(dim=1, keepdim=True)
        # compute mask
        static_mask = (diff > mask_threshold).float()
        # return
        return static_mask
    
    def compute_disp_losses(self, inputs, outputs):
        loss_dict = {}
        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]
            target = self.get_color_input(inputs, 0, 0)
            reprojection_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            use_static_mask = self.opt.use_static_mask
            # update ego diff
            if use_static_mask:
                with torch.no_grad():
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = self.get_color_input(inputs, frame_id, 0)

                        # get diff of two frames
                        diff = (pred - target).abs().mean(dim=1)
                        diff = torch.flatten(diff, 1)

                        # compute quantile
                        quantile = torch.quantile(diff, self.opt.static_mask_quantile, dim=1)
                        mean_quantile = quantile.mean()

                        # update
                        self.ego_diff.update(mean_quantile)

            # compute mask
            for frame_id in self.opt.frame_ids[1:]:
                pred = self.get_color_input(inputs, frame_id, 0)
                color_diff = self.compute_reprojection_loss(pred, target)
                identity_reprojection_loss = color_diff + torch.randn(color_diff.shape).type_as(color_diff) * 1e-5

                # static mask
                if use_static_mask:
                    static_mask = self.get_static_mask(pred, target)
                    identity_reprojection_loss *= static_mask
                    identity_reprojection_loss *= inputs[("light_mask", 0, 0)]

                reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reproject_loss = self.compute_reprojection_loss(pred, target)
                reproject_loss *= inputs[("light_mask", 0, 0)]
                reprojection_losses.append(reproject_loss)
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = get_smooth_loss(disp, self.get_color_input(inputs, 0, scale))
            loss_dict[('smooth_loss', scale)] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict
