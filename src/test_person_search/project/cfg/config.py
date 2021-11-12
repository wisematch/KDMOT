import os
import torch
import argparse
import datetime
import shutil
from src.test_person_search.project.engine.getengine import get_config_setter


class BaseConfig(object):
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #########
        # Train #
        #########
        parser.add_argument('--double_bias', action='store_true',
                            help='Whether to double the learning rate for bias')
        parser.add_argument('--bias_decay', action='store_true',
                            help='Whether to have weight decay on bias as well')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--no_flip', action='store_true',
                            help='Use horizontally-flipped images during training?')
        ########
        # Test #
        ########
        parser.add_argument('--gallery_size', type=int, default=100, choices=[-1, 50, 100, 500, 1000, 2000, 4000],
                            help='gallery size for evaluation, -1 for full set')
        parser.add_argument('--eval_batch_size', default=4, type=int, help='Test stage Batch size')
        parser.add_argument('--eval_gt', action='store_true', help='Test using ground truth bounding-box')
        parser.add_argument('--cws', action='store_true')

        ########
        # Data #
        ########
        parser.add_argument('--data_root', default='/export/wei.zhang/datasets/prw/PRW-v16.04.20', type=str)
        parser.add_argument('--cache_dir', default='./cache/', type=str)
        parser.add_argument('--n_batch_per_epoch', default=0, type=int)
        parser.add_argument('--div', action='store_true', help='Input image divides 255.')
        parser.add_argument('--RGB', action='store_true', help='Load image in RGB order for PyTorch pre-trained model.')
        now = datetime.datetime.now()
        parser.add_argument('--ckpt_dir', type=str, help='models are saved here',
                            default=os.path.join('/export/wei.zhang/PycharmProjects/kd_person_search/checkpoints/{}'.format(now.strftime("%Y%m%d"))))
        ########
        # Misc #
        ########
        parser.add_argument('--model_verbose', action='store_true', help='Print architecture of model')
        parser.add_argument("-dis", '--display', action='store_true')
        parser.add_argument('--seed', default=7, type=int)
        parser.add_argument('--eps', default=1e-14, type=float, help="A small number that's used many times")
        parser.add_argument('--device', default='cuda', help='device')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.backbone
        model_option_setter = get_config_setter(model_name)
        parser = model_option_setter(parser, opt.is_test)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults
        self.parser = parser

        return parser.parse_args()

    def print_options(self, cfg):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(cfg).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        cfg.expr_dir = os.path.join(cfg.ckpt_dir, cfg.expr_name)
        if not os.path.exists(cfg.expr_dir):
            os.makedirs(cfg.expr_dir)
            # print('\nmake dir {}'.format(cfg.expr_dir))
        # shutil.copy("./run.sh", cfg.expr_dir)
        file_name = os.path.join(cfg.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        cfg = self.gather_options()
        # process opt.suffix
        if cfg.suffix:
            suffix = ('*' + cfg.suffix.format(**vars(cfg))) if cfg.suffix != '' else ''
            cfg.expr_name = cfg.expr_name + suffix

        if cfg.is_test:
            cfg.expr_name = 'test'

        self.print_options(cfg)

        # set gpu ids
        str_ids = cfg.gpu_ids.split(',')
        cfg.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                cfg.gpu_ids.append(id)

        if len(cfg.gpu_ids) > 1:
            cfg.parallel = True

        self.cfg = cfg
        return self.cfg


class Config(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)
        parser.add_argument('--benchmark', type=str, default='prw')
        parser.add_argument('--backbone', type=str, default='bsl')
        parser.add_argument('--bg_aug', action='store_true', help="background augmentation")
        parser.add_argument('--clip_grad', action='store_true')
        parser.add_argument('--conv5_stride', type=int, default=2)
        parser.add_argument('--continue_train', action='store_true', help=
        'set current epoch to ckpt.epoch, current best_rank1 to ckpt.best_rank1, if continue training')


        parser.add_argument('--display_freq', default=5, type=int)
        parser.add_argument('--distributed', action='store_true')
        parser.add_argument('--logger', default=None, type=object)

        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--lr_factor_teacher', type=float, default=1.0, help="lr factor for re-ID model")
        parser.add_argument('--lr_policy', type=str, default='mlt_step',
                            help='learning rate policy: lambda|step|plateau|cosine|mlt_step')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
        parser.add_argument("-ldm", '--lr_decay_milestones', default=[4], nargs='+', type=int,
                            help="multiply by a gamma every lr_decay_milestones iterations")
        parser.add_argument('--weight_decay', type=float, default=0.0001)  # 5e-4
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--eval_epoch', default=[], nargs='+', type=int)
        parser.add_argument('--max_epoch', default=5, type=int)
        parser.add_argument('--num_train_ids', type=int, default=0)
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--warmup_epochs', type=int, default=-1)
        parser.add_argument('--warmup_mode', type=str, default="linear", help="linear|constant")

        parser.add_argument('--expr_dir', type=str, default='',
                            help='ckpt_dir+expr_name')
        parser.add_argument('--probe_type', default='origin', type=str, help="origin|mini|rand")

        parser.add_argument('--n_worker', default=16, type=int)

        parser.add_argument('--n_instance', default=2, type=int,
                            help='number instances per pid')

        parser.add_argument('--id_sampler', action='store_true',
                            help='whether randomly sample identity')
        parser.add_argument('--freeze_top_bn', action='store_true',
                            help='whether fix batch_norm in the top layer of rcnn')

        # config optimization
        parser.add_argument('--optim', default="sgd", type=str, help='training optimizer')

        # resume trained model
        parser.add_argument('--load_ckpt', type=str, default='')

        parser.add_argument('--is_test', action='store_true', help='training or test')

        parser.add_argument('--expr_name', type=str, default='test',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--num_class', type=int, default=0, help='#training classes')
        parser.add_argument('--input_size', type=int, default=[256, 128], nargs='+',
                            help='training stage input image size [height width]')

        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--transforms', default=['resize'], nargs='+', type=str,
                            help="transforms that would be applied to the input image")

        # distributed training parameters
        parser.add_argument('--aspect-ratio-group-factor', default=-1, type=int)
        parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

        return parser
