# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
from pathlib import Path
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric, UF1, LossMetric, UAS, discoF1

class Train(CMD):
    def __call__(self, args):
        self.args = args
        self.device = args.device

        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        create_save_path(args)
        log = get_logger(args)

        self.save_dir = args.save_dir

        self.optimizer = get_optimizer(args.optimizer, self.model)


        log.info("Create the model")
        log.info(f"{self.model}\n")
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        log.info(self.optimizer)
        log.info(args)
        eval_loader = dataset.val_dataloader
        test_loader = dataset.test_dataloader
        '''
        Training
        '''
        train_arg = args.train
        self.train_arg = train_arg
        self.dataset = dataset


        # print("Evaluating.")
        # eval_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        # dev_f1_metric, dev_ll = self.evaluate(eval_loader_autodevice, test=False, save_dir=self.args.save_dir)
        # log.info(f"{'dev f1:':6}   {dev_f1_metric}")
        # log.info(f"{'dev ll:':6}   {dev_ll}")


        for epoch in range(1, train_arg.max_epoch + 1):
            '''
            Auto .to(self.device)
            '''
            # curriculum learning. Used in compound PCFG.
            if train_arg.curriculum:
                train_loader = dataset.train_dataloader(max_len=min(train_arg.start_len + (epoch-1)*(train_arg.increment - 1), train_arg.max_len), epoch=epoch)
            else:
                train_loader = dataset.train_dataloader(max_len=train_arg.max_len, epoch=epoch)


            train_loader_autodevice = DataPrefetcher(train_loader, device=self.device)
            eval_loader_autodevice = DataPrefetcher(eval_loader, device=self.device)
            test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
            start = datetime.now()
            self.train(train_loader_autodevice, epoch=epoch)
            log.info(f"Epoch {epoch} / {train_arg.max_epoch}:")

            dev_f1_metric, dev_ll = self.evaluate(eval_loader_autodevice, save_dir=self.save_dir)
            log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            log.info(f"{'dev ll:':6}   {dev_ll}")
            #

            if train_arg.eval_test:
                test_f1_metric, test_ll = self.evaluate(test_loader_autodevice, save_dir=self.save_dir)
                log.info(f"{'test f1:':6}   {test_f1_metric}")
                log.info(f"{'test ll:':6}   {test_ll}")
            #

            t = datetime.now() - start

            # save the model if it is the best so far
            if dev_f1_metric > best_metric:
                best_metric = dev_f1_metric
                best_e = epoch
                torch.save(
                    obj=self.model.state_dict(),
                    f=args.save_dir + "/best.pt"
                )
                log.info(f"{t}s elapsed, Best saved\n")
            else:
                log.info(f"{t}s elapsed\n")

            total_time += t
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                break


    def train(self, loader, epoch):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)
        train_arg = self.args.train
        for x, _ in t:
            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            assert (~torch.isnan(loss)).all()
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         train_arg.clip)
            self.optimizer.step()
            t.set_postfix(loss=loss.item())
        return






