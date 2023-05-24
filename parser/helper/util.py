import time
import os
import logging
from distutils.dir_util import copy_tree

from parser.model.N_LCFRS import NeuralLCFRS
from parser.model.TN_LCFRS import TNLCFRS
from parser.model.C_LCFRS import CompoundLCFRS

import torch



def get_model(args, dataset):
    if args.model_name == 'TN_LCFRS':
        return TNLCFRS(args, dataset).to(dataset.device)


    elif args.model_name == 'N_LCFRS':
        return NeuralLCFRS(args, dataset).to(dataset.device)

    elif args.model_name == 'C_LCFRS':
        return CompoundLCFRS(args, dataset).to(dataset.device)

    else:
        raise KeyError


def get_optimizer(args, model, scale=1):
    if args.name == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr * scale, betas=(args.mu, args.nu))
    elif args.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=args.lr * scale, betas=(args.mu, args.nu), weight_decay=args.weight_decay)
    elif args.name == 'sgd':
        return torch.optim.SGD(params=model.parameters(), lr=args.lr * scale, momentum=.9)
    else:
        raise NotImplementedError

def get_logger(args, log_name='train',path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                             time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

