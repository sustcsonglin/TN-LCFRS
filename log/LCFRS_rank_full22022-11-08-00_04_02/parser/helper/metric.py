# -*- coding: utf-8 -*-
import pdb
from collections import Counter, defaultdict
import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return -1e9

class UF1(Metric):
    def __init__(self, eps=1e-8, device=torch.device("cuda")):
        super(UF1, self).__init__()
        self.f1 = 0.0
        self.evalb = 0.0
        self.n = 0.0
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.device = device


    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            # in the case of sentence length=1
            if len(pred) == 0:
                continue
            length = max(gold,key=lambda x:x[1])[1]
            #removing the trival span
            gold = list(filter(lambda x: x[0]+1 != x[1], gold))
            pred = list(filter(lambda x: x[0]+1 != x[1], pred))
            #remove the entire sentence span.
            gold = list(filter(lambda x: not (x[0]==0 and x[1]==length), gold))
            pred = list(filter(lambda x: not (x[0]==0 and x[1]==length), pred))
            #remove label.
            gold = [g[:2] for g in gold]
            pred = [p[:2] for p in pred]
            gold = list(map(tuple, gold))
            #corpus f1
            for span in pred:
                if span in gold:
                    self.tp += 1
                else:
                    self.fp += 1
            for span in gold:
                if span not in pred:
                    self.fn += 1

            #sentence f1
            #remove duplicated span.
            gold = set(gold)
            pred = set(pred)
            overlap = pred.intersection(gold)
            prec = float(len(overlap)) / (len(pred) + self.eps)
            reca = float(len(overlap)) / (len(gold) + self.eps)
            if len(gold) == 0:
                reca = 1.
                if len(pred) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            self.f1 += f1
            self.n += 1

    @property
    def sentence_uf1(self):
        return self.f1 / self.n

    @property
    def corpus_uf1(self):
        if self.tp == 0 and self.fp == 0:
            return 0

        prec = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return corpus_f1

    @property
    def score(self):
        return self.sentence_uf1

    def __repr__(self):
        s = f"Sentence F1: {self.sentence_uf1:6.2%} Corpus F1: {self.corpus_uf1:6.2%} "
        return s



# TOOD:
class discoF1(Metric):
    def __init__(self, eps=1e-8, device=torch.device("cuda")):
        super(discoF1, self).__init__()
        self.f1 = 0.0
        self.evalb = 0.0
        self.n = 0.0
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.device = device

        self.tp_dic = 0.0
        self.fp_dic = 0.0
        self.n_dic = 0.0
        self.fn_dic = 0.0


    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            # in the case of sentence length=1
            # if len(pred) == 0:
            #     continue
            gold, gold_discont = gold
            pred, pred_discont = pred
            length = max(gold,key=lambda x:x[1])[1]
            #removing the trival span
            gold = list(filter(lambda x: x[0]+1 != x[1], gold))
            pred = list(filter(lambda x: x[0]+1 != x[1], pred))
            #remove the entire sentence span.
            gold = list(filter(lambda x: not (x[0]==0 and x[1]==length), gold))
            pred = list(filter(lambda x: not (x[0]==0 and x[1]==length), pred))
            #remove label.

            original_gold = gold

            gold = [g[:2] for g in gold]
            pred = [p[:2] for p in pred]

            # stupid way.
            gold_discont = [g[:1] for g in gold_discont]
            gold = list(map(tuple, gold))
            gold_discont2 = []

            for span in gold_discont:
                if len(span[0]) > 2:
                    continue
                candidate = []
                for d_ in span[0]:
                    candidate += d_
                gold_discont2.append(candidate)
            gold_discont= gold_discont2
            gold_discont = list(map(tuple, gold_discont))

            #corpus f1
            for span in pred:
                if span in gold:
                    self.tp += 1
                else:
                    self.fp += 1
            for span in gold:
                if span not in pred:
                    self.fn += 1
            #sentence f1
            #remove duplicated span.
            gold = set(gold)
            pred = set(pred)
            overlap = pred.intersection(gold)
            prec = float(len(overlap)) / (len(pred) + self.eps)
            reca = float(len(overlap)) / (len(gold) + self.eps)
            if len(gold) == 0:
                reca = 1.
                if len(pred) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            self.f1 += f1
            self.n += 1

            for span in pred_discont:
                if span in gold_discont:
                    self.tp_dic += 1
                    print("hello", span)

                else:
                    self.fp_dic += 1

            for span in gold_discont:
                if span not in pred_discont:
                    self.fn_dic += 1





    @property
    def sentence_uf1(self):
        return self.f1 / self.n

    @property
    def corpus_uf1(self):
        if self.tp == 0 and self.fp == 0:
            return 0

        prec = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return  corpus_f1, prec, recall

    @property
    def corpus_uf1_disco(self):
        if self.tp_dic == 0 and self.fp_dic == 0:
            return 0, 0, 0
        prec = self.tp_dic / (self.tp_dic + self.fp_dic + 1e-9)
        recall = self.tp_dic / (self.tp_dic + self.fn_dic + 1e-9)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return corpus_f1, prec, recall

    @property
    def df1(self):
        if self.tp_dic == 0 and self.fp_dic == 0:
            return 0, 0, 0
        prec = self.tp_dic / (self.tp_dic + self.fp_dic + 1e-9)
        recall = self.tp_dic / (self.tp_dic + self.fn_dic + 1e-9)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return corpus_f1


    @property
    def all_uf1(self):
        if self.tp == 0 and self.fp == 0:
            return 0, 0, 0
        tp = self.tp_dic + self.tp
        fp = self.fp_dic + self.fp
        fn = self.fn + self.fn_dic
        prec = tp/(tp+fp+1e-9)
        recall = tp/(tp+fn+1e-9)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return corpus_f1

    @property
    def score(self):
        return self.df1

    def __repr__(self):
        f1, prec, recall = self.corpus_uf1_disco
        f1_c, prec_c, recall_c = self.corpus_uf1
        s = f"F1:{self.all_uf1:6.2%} Co F1: {f1_c:6.2%}, Disco F1: {f1:6.2%}, Co P: {prec_c:6.2%}, Co R:{recall_c:6.2%},  Disco P:{prec:6.2%}, Disco R:{recall:6.2%},   Cont TP: {self.tp}, Cont FP: {self.fp}, Cont FN: {self.fn}, Disco TP: {self.tp_dic}, Disco FP: {self.fp_dic}, Disco FN: {self.fn_dic}"
        # s = f"F1:{self.all_uf1:6.2%} Co F1: {self.corpus_uf1:6.2%}, Disco F1: {f1:6.2%}"
        return s


class printTree():
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        self.pred_trees = []
        self.raw_trees = []


    def __call__(self, p_tree, raw_tree):
        self.pred_trees += p_tree
        self.raw_trees += raw_tree


    def print_tree(self, test):
        mode = 'test' if test else 'val'

        with open(f"{self.save_dir}/raw_{mode}.discbracket", 'w', encoding='utf8') as f:
            for raw_tree in self.raw_trees:
                f.write(raw_tree.strip())
                f.write('\n')

        with open(f"{self.save_dir}/pred_{mode}.discbracket", 'w', encoding='utf8') as f:
            for pred_tree in self.pred_trees:
                f.write(pred_tree)
                f.write('\n')



class UAS(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.direct_correct = 0.0
        self.undirect_correct = 0.0
        self.total_sentence = 0.0
        self.correct_root = 0.0

    @property
    def score(self):
        return   self.direct_correct / self.total

    def __call__(self, predicted_arcs, gold_arcs):

        for pred, gold in zip(predicted_arcs, gold_arcs):
            assert len(pred) == len(gold)

            if len(pred) > 0:
                self.total_sentence+=1.

            for (head, child) in pred:
                if gold[int(child)] == int(head) + 1:
                    self.direct_correct += 1.
                    self.undirect_correct += 1.
                    if int(head) + 1 == 0:
                        self.correct_root += 1.

                elif gold[int(head)] == int(child) + 1:
                    self.undirect_correct += 1.

                self.total += 1.


    def __repr__(self):
        return "UDAS: {}, UUAS:{}, root:{} ".format(self.score, self.undirect_correct/self.total, self.correct_root/self.total_sentence)


class LossMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_kl = 0.0
        self.calling_time = 0


    def __call__(self, likelihood):
        self.calling_time += 1
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()

    @property
    def avg_loss(self):
        return self.total_likelihood / self.total


    def __repr__(self):
        return "avg likelihood: {} kl: {}, total likelihood:{}, n:{}".format(self.avg_likelihood,self.avg_kl,self.total_likelihood, self.total)

    @property
    def score(self):
        return (self.avg_likelihood + self.avg_kl).item()


class LikelihoodMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_word = 0

    @property
    def score(self):
        return self.avg_likelihood


    def __call__(self, likelihood, lens):
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()
        # Follow Yoon Kim
        self.total_word += (lens.sum() + lens.shape[0])

    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.total


    @property
    def perplexity(self):
        return (-self.total_likelihood / self.total_word).exp()

    def __repr__(self):
        return "avg likelihood: {}, perp. :{}".format(self.avg_likelihood, self.perplexity)



