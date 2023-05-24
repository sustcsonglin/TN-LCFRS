# -*- coding: utf-8 -*-
import os
import pdb

import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS, discoF1, printTree

import time



class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
        metric_ll = LikelihoodMetric()
        
        for x, _ in t:
            self.optimizer.zero_grad()
            partition = self.model.loss(x)
            try:
                assert (~torch.isnan(partition)).all()
            except:
                print("?????")
                pdb.set_trace()
                
            loss = -partition.mean()
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            try:
                metric_ll(partition, x['seq_len'])
            except:
                print("?")
                pdb.set_trace()

            t.set_postfix(loss=loss.item(), ppl=metric_ll)

        return metric_ll

    @torch.no_grad()
    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None, save_dir=None, test=False):
        if model == None:
            model = self.model
        model.eval()
        metric_f1 = discoF1()
        # if eval_dep:
        #     metric_uas = UAS()
        
        print_tree =  printTree(save_dir=save_dir)
        metric_ll = LikelihoodMetric()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))
        for x, y in t:
            try:
                result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
                metric_f1(result['prediction'], y['gold_tree'])
                
                metric_ll(result['partition'], x['seq_len'])
                print_tree(result['prediction_tree'], x['raw_tree'])
            except:
                pdb.set_trace()


        print_tree.print_tree(test=test)

        if not eval_dep:
            return metric_f1, metric_ll
        else:
            return metric_f1, metric_uas, metric_ll




