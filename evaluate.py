# -*- coding: utf-8 -*-

from ast import arg
import os
from parser.cmds.evaluate import Evaluate
import torch
from easydict import EasyDict as edict
import yaml
import click



@click.command()
@click.option("--device", '-d', default='0')
@click.option("--load_from_dir", default="")
@click.option("--test_file", default="")
def main(device, load_from_dir, test_file):
    print("Loading from:", load_from_dir)
    yaml_cfg = yaml.safe_load(open(load_from_dir + "/config.yaml", 'r'))
    print("Successfully load.")
    print(yaml_cfg)
    print("Test file:", test_file)
    args = edict(yaml_cfg)
    args.device = device
    args.load_from_dir = load_from_dir
    args.data.test_file = test_file
    print(f"Set the device with ID {args.device} visible")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    command = Evaluate()
    command(args)


if __name__ == '__main__':
    main()