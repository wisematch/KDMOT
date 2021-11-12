from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import time
import pickle
import shutil
import inspect
import datetime
import torch
import numpy as np
from project.data.preproc import create_data
from project.data import getitem
from ..lib.models.model import create_model, load_model
from .project.misc import util as util
import torch.nn.functional as F
from ..lib.models.decode import mot_decode
from ..lib.models.utils import _tranpose_and_gather_feat
from ..lib.utils.post_process import ctdet_post_process
from ..lib.models.teacher_model import create_teacher_model, load_teacher_model
from ..lib.opts import opts


def eval_person_search(opt):
    # data_cfg = opt.data_cfg
    # f = open(data_cfg)
    # data_cfg_dict = json.load(f)
    # f.close()
    #
    # test_paths = data_cfg_dict['test']
    # dataset_root = data_cfg_dict['root']
    # opt.device = torch.device('cuda')
    cfg = opt.cfg

    imdb = create_data(cfg.benchmark, cfg.data_root, True, training=False, probe_type=cfg.probe_type)
    roidb = imdb.roidb
    print('{:d} roidb entries'.format(len(roidb)))  # number of training/test images

    probe_getitem = getitem.ProbeGetitem(imdb.probes)
    gallery_getitem = getitem.SequentialGetitem(roidb, imdb.num_classes, div=cfg.div, BGR=not cfg.RGB)
    probe_loader = torch.utils.data.DataLoader(probe_getitem, batch_size=1,
                                               num_workers=cfg.n_worker, collate_fn=collate_fn())
    gallery_loader = torch.utils.data.DataLoader(gallery_getitem, batch_size=cfg.eval_batch_size,
                                                 num_workers=cfg.n_worker, collate_fn=collate_fn())

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()


    # dir for saving features and detections
    output_dir = os.path.join(imdb.cache_path, imdb.name, 'detections')  # ./cache/benchmark_name/features
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----------------------------------------------------------------------------------------------------------------
    # Traversing gallery images
    def extract_gallery():
        num_images = len(imdb.image_names)
        gallery_boxes = []  # len(gallery_boxes)=num_classes
        gallery_features = []

        _t = {'im_detect': util.Timer(), 'reid': util.Timer()}

        print("\nextracting gallery features ... ")
        for idx, data in enumerate(gallery_loader):

            images, targets = data['images'], data['targets']
            inp_img = list(img.to(torch.device('cuda')) for img in images)

            torch.cuda.synchronize()

            # 用teacher提取query的embedding

            _t['im_detect'].tic()
            # rois, _ = engine.forward()
            output = model(images.cuda())[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg'] if opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

            width = img0.shape[1]
            height = img0.shape[0]
            inp_height = im_blob.shape[2]
            inp_width = im_blob.shape[3]

            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
            meta = {'c': c, 's': s,
                    'out_height': inp_height // opt.down_ratio,
                    'out_width': inp_width // opt.down_ratio}

            def post_process(dets, meta):
                dets = dets.detach().cpu().numpy()
                dets = dets.reshape(1, -1, dets.shape[2])
                dets = ctdet_post_process(
                    dets.copy(), [meta['c']], [meta['s']],
                    meta['out_height'], meta['out_width'], opt.num_classes)
                for j in range(1, opt.num_classes + 1):
                    dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
                return dets[0]

            def merge_outputs(detections):
                results = {}
                for j in range(1, opt.num_classes + 1):
                    results[j] = np.concatenate(
                        [detection[j] for detection in detections], axis=0).astype(np.float32)

                scores = np.hstack(
                    [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
                # if len(scores) > self.max_per_image:
                #     kth = len(scores) - self.max_per_image
                #     thresh = np.partition(scores, kth)[kth]
                #     for j in range(1, opt.num_classes + 1):
                #         keep_inds = (results[j][:, 4] >= thresh)
                #         results[j] = results[j][keep_inds]
                return results

            dets = post_process(dets, meta)
            dets = merge_outputs([dets])[1]

            _t['im_detect'].toc()

            # box_and_prob = torch.cat([roi['boxes'], roi['scores'][:, None]], dim=1)
            # gallery_boxes.append(box_and_prob.cpu().numpy())
            # gallery_features.append(roi['feats'].cpu().numpy())

            gallery_boxes.append(dets.cpu().numpy())
            gallery_features.append(id_feature.cpu().numpy())

            # if cfg.display:
            #     im = cv2.imread(imdb.image_path_at(i))
            #     im2show = np.copy(im)
            # if cfg.display:
            #     im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.0)
            # misc_tic = time.time()
            # misc_toc = time.time()
            # nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(
                idx + 1, len(gallery_loader) + 1, _t['im_detect'].average_time, _t['reid'].average_time))
            sys.stdout.flush()

            # if cfg.display:
            #     cv2.imwrite(os.path.join(output_dir, 'gallery.png'), im2show)

        print('total time:{:.3f}s, reid time:{:.3f}s'.format(_t['im_detect'].average_time, _t['reid'].average_time))

        assert len(gallery_features) == len(gallery_boxes) == num_images

        with open(os.path.join(output_dir, 'gallery_features.pkl'), 'wb') as f:
            pickle.dump(gallery_features, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(output_dir, 'gallery_boxes.pkl'), 'wb') as f:
            pickle.dump(gallery_boxes, f, pickle.HIGHEST_PROTOCOL)
        return gallery_boxes, gallery_features

        # ----------------------------------------------------------------------------------------------------------------
        # Traversing probe images
    def extract_probe():
        num_probe_ims = len(imdb.probes)
        probe_features = {'feat': [0 for _ in range(num_probe_ims)]}
        _t = {'im_exfeat': util.Timer(), 'misc': util.Timer()}

        print("\nextracting probe features ... ")
        for i, data in enumerate(probe_loader):
            images, target = data['images'], data['targets']
            image = list(img.to(torch.device('cuda')) for img in images)
            target = [{"boxes": t.view(-1, 4).to(torch.device('cuda'))} for t in target]
            torch.cuda.synchronize()
            assert len(image) == 1, "Only support single image input"

            _t['im_exfeat'].tic()

            # load teacher
            teacher_model = create_teacher_model()
            teacher_model = load_teacher_model(teacher_model, opt.model_t_path)


            # feat = engine.extra_box_feat(image, target)
            _t['im_exfeat'].toc()

            probe_features['feat'][i] = feat[0].cpu().numpy()

            # print('im_exfeat: {:d}/{:d} {:.3f}s'.format(i + 1, num_probe_ims, _t['im_exfeat'].average_time))
            sys.stdout.write('im_exfeat: {:d}/{:d} {:.3f}s   \r' \
                             .format(i + 1, num_probe_ims, _t['im_exfeat'].average_time))
            sys.stdout.flush()

            # if cfg.display:
            #     im2show = vis_detections(np.copy(im), 'person', np.concatenate((roi, np.array([[1.0]])), axis=1), 0.3)
            #     cv2.imwrite(os.path.join(output_dir, 'probe.png'), im2show)

        with open(os.path.join(output_dir, 'probe_features.pkl'), 'wb') as f:
            pickle.dump(probe_features, f, pickle.HIGHEST_PROTOCOL)

    start = time.time()
    flag = False
    if flag:
        # load pickle file for debugging evaluation code
        with open(os.path.join(output_dir, 'probe_features.pkl'), 'rb') as f:
            probe_features = pickle.load(f)
        with open(os.path.join(output_dir, 'gallery_features.pkl'), 'rb') as f:
            gallery_features = pickle.load(f)
        with open(os.path.join(output_dir, 'gallery_boxes.pkl'), 'rb') as f:
            gallery_boxes = pickle.load(f)
    else:
        with torch.no_grad():
            probe_features = extract_probe()
            gallery_boxes, gallery_features = extract_gallery()


    return probe_features

    # ----------------------------------------------------------------------------------------------------------------
    # evaluate pedestrian detection and person search
    logger.msg('\nevaluating detections')

    # all detection
    # todo-bug: det_thr 0.5 to 0.05 for single-stage detector
    ap, recall = imdb.evaluate_detections(gallery_boxes, score_thresh=cfg.det_thr)
    logger.msg('all detection:')
    logger.msg('  recall = {:.2%}'.format(recall))
    logger.msg('  ap = {:.2%}'.format(ap))

    # labeled only detection
    ap, recall = imdb.evaluate_detections(gallery_boxes, score_thresh=cfg.det_thr, labeled_only=True)
    logger.msg('labeled_only detection:')
    logger.msg('  recall = {:.2%}'.format(recall))  # todo(note): added 20200609
    logger.msg('  ap = {:.2%}'.format(ap))

    # evaluate search
    kwargs = dict(score_thresh=0.5, base_iou_thresh=0.5, dump_json=os.path.join(cfg.expr_dir, 'results.json'))
    if "ssm" == cfg.benchmark: kwargs.update({"gallery_size": cfg.gallery_size})
    ap, topk, accs, recall = imdb.evaluate_search(gallery_boxes, gallery_features, probe_features['feat'], **kwargs)

    logger.msg('search ranking:')
    logger.msg('  recall = {:.2%}'.format(recall))  # does the true_positive of a probe been detected.
    logger.msg('  mAP = {:.2%}'.format(ap))
    for i, k in enumerate(topk):
        logger.msg('  top-{:2d} = {:.2%}'.format(k, accs[i]))

    end = time.time()
    print("test time: %0.1fmin" % ((end - start) / 60))




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    with torch.no_grad():
        tpr = eval_person_search(opt)