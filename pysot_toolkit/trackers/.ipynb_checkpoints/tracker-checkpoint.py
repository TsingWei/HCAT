from __future__ import absolute_import

import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time
# from torch2trt import torch2trt

class Tracker(object):

    def __init__(self, name, net, feature_size, window_penalty=0.49, penalty_k=0, exemplar_size=128, instance_size=256):
        self.name = name
        self.net = net
        self.window_penalty = window_penalty
        self.penalty_k = penalty_k
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.feature_size = feature_size

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        # tic = time.time()
        resize_factor = original_sz / model_sz

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1

        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        r, c, k = im.shape
        im_context = im[max(0,int(context_ymin)):min(int(context_ymax)+1,r),max(0,int(context_xmin)):min(int(context_xmax)+1,c),:]
        r_context, c_context, _ = im_context.shape
        r_context_resize = round(r_context / resize_factor)
        c_context_resize = round(c_context / resize_factor)
        left_pad = round(left_pad / resize_factor)
        top_pad = round(top_pad / resize_factor)
        right_pad = round(right_pad / resize_factor)
        bottom_pad = round(bottom_pad / resize_factor)

        im_patch_context = cv2.resize(im_context, (c_context_resize, r_context_resize))

        te_im = np.zeros([model_sz,model_sz,k])
        te_im[top_pad:top_pad + r_context_resize, left_pad:left_pad + c_context_resize, :] = im_patch_context

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c_context_resize, :] = avg_chans
        if bottom_pad:
            te_im[r_context_resize + top_pad:, left_pad:left_pad + c_context_resize, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c_context_resize + left_pad:, :] = avg_chans
        # print(time.time()-tic)


        # if not np.array_equal(model_sz, original_sz):
        #     im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = te_im
        _,r,c = im_patch.shape
        if not r == c == model_sz:
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))


        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)


        im_patch = im_patch.cuda()

        # 0.3ms
        return im_patch



    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()
        hanning = np.hanning(self.feature_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break


        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        self.mean_ = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
        self.std_ = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        z_crop /= 255
        z_crop -= self.mean_
        z_crop /= self.std_
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        # self.inplace = False
        # z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        # z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        # print(z_crop.shape)
        # torch.Size([1, 3, 64, 64])
        self.net.template(z_crop, self.feature_size)
        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        # calculate x crop size
        tic_all = time.time()
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))
        # print('sqrt_time' + str(time.time() - tic))
        # 0.04ms


        # get crop
        # tic = time.time()
        x_crop = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)
        # print('window_time'+str(time.time()-tic))
        #0.9 ms

        # normalize
        # tic = time.time()
        # x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        # x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
        x_crop /= 255
        x_crop -= self.mean_
        x_crop /= self.std_
        # print('norm_time'+str(time.time()-tic))
        # 0.06ms

        # track
        # tic = time.time()
        # x_crop.shape
        # torch.Size([1, 3, 96, 96])
        tic_model = time.time()
        outputs = self.net.track(x_crop)
        model_time = time.time()-tic_model
        # print('track_time'+str(time.time()-tic))
        #4.2ms
        # print('---------------------------------------')
        # tic = time.time()
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])
        # print('convert_time' + str(time.time() - tic))
        # 0.14ms

        # tic = time.time()
        pscore = score
        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        all_time = time.time()-tic_all

        out = {'target_bbox': bbox,
               'best_score': pscore[best_idx],
               'model_time':model_time,
               'all_time':all_time}
        # print('last_time'+str(time.time()-tic))
        # 0.06ms
        return out

