import os
import numpy as np
from PIL import Image
from scipy.misc import imread

import torch
import torch.utils.data as data
from torchvision.ops.boxes import clip_boxes_to_image
from src.lib.datasets.dataset.jde import letterbox
import cv2
from torchvision.transforms import functional as F
import copy
import time

class SequentialGetitem(data.Dataset):

    def __init__(self, roidb, num_classes, div=False, BGR=True):
        self.roidb = roidb
        self._num_classes = num_classes
        self.div = True
        self.BGR = False
        self.width = 1088
        self.height = 608

    def get_height_and_width(self, index):
        return self.roidb[index]['height'], self.roidb[index]['width']

    def __getitem__(self, index):
        single_item = self.roidb[index]
        # Image
        im = cv2.imread(single_item['img_path'])  # RGB, HWC, 0-255
        im_PIL = Image.open(single_item['img_path']).convert('RGB') # RGB, HWC, 0-255

        height = im.shape[0]
        width = im.shape[1]

        img, _, _, _ = letterbox(im, height=self.height, width=self.width)

        # im = np.array(Image.open(single_item['img_path']).convert('RGB'))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0


        if single_item['flipped']:
            img = img[:, ::-1, :]
        # im = im.astype(np.float32, copy=False)
        # image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW
        image = torch.from_numpy(img)

        # Targets
        gt_boxes = single_item['boxes'].astype(np.int32, copy=False)
        gt_boxes = torch.from_numpy(gt_boxes).float()  # TODO(BUG): dtype
        # clip boxes of which coordinates out of the image resolution
        crop_boxes = copy.deepcopy(gt_boxes)

        gt_boxes = clip_boxes_to_image(gt_boxes, tuple(img.shape[1:]))
        crop_boxes = clip_boxes_to_image(crop_boxes, tuple(im.shape[:2]))
        crop_boxes = crop_boxes.numpy()

        crop_ims = []
        for id in range(crop_boxes.shape[0]):

            box = crop_boxes

            x = box[id][0]
            y = box[id][1]
            h = box[id][2] - box[id][0]
            w = box[id][3] - box[id][1]

            crop_im = F.resized_crop(copy.deepcopy(im_PIL), int(y), int(x), int(w), int(h), size=[256, 128])

            save_crop_im = False
            if save_crop_im:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('Agg')
                # plt.figure(figsize=(50, 50))

                plt.imshow(crop_im)

                # plt.imshow(im_PIL)
                # labels_ = box.copy()
                # plt.plot(labels_[:, [0, 2, 2, 0, 0]].T, labels_[:, [1, 1, 3, 3, 1]].T, '.-')

                plt.axis('off')

                plt.savefig('/export/wei.zhang/vis_debug/' + str(index) + '_' + str(id) + '.jpg')
                # cv2.imwrite('/export/wei.zhang/vis_debug/' + str(index) + '_' + str(h) + '.jpg', im)


                # im.save('/export/wei.zhang/vis_debug/' + str(index) + '_' + im_name.split('/')[-1])
                # time.sleep(2)
                # crop_im2.save('/export/wei.zhang/PycharmProjects/FairMOT/vis_debug/' + '2_'+ str(k) + '_' + img_path.split('/')[-1])
                time.sleep(2)
                assert 0

            crop_im = np.array(crop_im).transpose((2, 0, 1))
            crop_ims.append(torch.from_numpy(crop_im))

        target = dict(
            boxes=gt_boxes,  # (num_boxes 4)
            labels=torch.from_numpy(single_item['gt_classes']).int(),
            pids=torch.from_numpy(single_item['gt_pids']).long(),
            img_name=single_item['img_name'],
        )
        if 'mask_path' in single_item:
            # foreground mask
            mask = imread(single_item['mask_path']).astype(np.int32, copy=False)  # (h w) in {0,255}
            assert np.ndim(mask) == 2
            if single_item['flipped']: mask = mask[:, ::-1]
            target['mask'] = torch.from_numpy(mask.copy())[None] / 255.  # 3D tensor(1HW) in {0,1}

        item = dict(image=image,
                    target=target,
                    height=height,
                    width=width,
                    im_PIL=im_PIL
                    )

        # visualization
        # util.plot_gt_on_img([image], [target], write_path=
        # "/home/caffe/code/deep-person-search/cache/img_with_gt_box/gt%d.jpg" % np.random.choice(list(range(10)), 1))

        return item

    def __len__(self):
        return len(self.roidb)


class ProbeGetitem(data.Dataset):
    def __init__(self, probe_list):
        self.probes = probe_list

    def __getitem__(self, index):
        outputs = self.probes[index]
        im_name, roi = outputs[0], outputs[1]
        # process input image
        # im = imread(im_name)  # RGB, HWC, 0-255

        im = Image.open(im_name).convert('RGB') # RGB, HWC, 0-255
        box = torch.from_numpy(roi.reshape(1, 4)).float()  # shape (1 4)


        x = box[0][0]
        y = box[0][1]
        h = box[0][2] - box[0][0]
        w = box[0][3] - box[0][1]

        im = F.resized_crop(copy.deepcopy(im), int(y), int(x), int(w), int(h), size=[256, 128])
        im = np.asarray(im)

        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2, rgb -> bgr
        # im = im[:, :, ::-1]  # TODO(NOTE): 加不加？RGB to BGR for caffe pretrained model
        im = im.astype(np.float32, copy=False)
        image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW

        save_crop_im = False
        if save_crop_im:
            cv2.imwrite('/export/wei.zhang/vis_debug/'+str(index)+'.jpg', im)
            # im.save('/export/wei.zhang/vis_debug/' + str(index) + '_' + im_name.split('/')[-1])
            # time.sleep(2)
            # crop_im2.save('/export/wei.zhang/PycharmProjects/FairMOT/vis_debug/' + '2_'+ str(k) + '_' + img_path.split('/')[-1])
            time.sleep(2)
            assert 0

        item = dict(image=image, target=box)
        return item

    def __len__(self):
        return len(self.probes)


class ListImageLoader(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = [os.path.join(self.path, p) for p in os.listdir(path)]

    def __getitem__(self, index):
        img_path = self.img_list[index]
        # read image
        im = imread(img_path)  # PIL image, RGB, HWC
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2, rgb -> bgr
        # im = im[:, :, ::-1]  # NOTE: RGB to BGR for caffe pretrained model

        im = im.astype(np.float32, copy=False)
        image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW
        return image

    def __len__(self):
        return len(self.img_list)
