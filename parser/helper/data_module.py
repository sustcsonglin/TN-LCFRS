import pdb
import pickle
from fastNLP.core.dataset import DataSet
from fastNLP.core.batch import DataSetIter
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.sampler import BucketSampler, ConstantTokenNumSampler
from torch.utils.data import Sampler
from collections import defaultdict
import os
import random
import numpy as np

class DataModule():
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.device = self.hparams.device
        self.setup()

    def prepare_data(self):
        pass

    def setup(self):
        print("Setup!")
        data = self.hparams.data
        train_dataset = DataSet()
        val_dataset = DataSet()
        test_dataset = DataSet()
        data.use_sup=False
        if data.use_sup:
            sup_dataset = DataSet()
        #
        word_vocab = Vocabulary(max_size=data.vocab_size)
        train_data =  pickle.load(open(data.train_file, 'rb'))
        
        val_data = pickle.load(open(data.val_file, 'rb'))
        test_data = pickle.load(open(data.test_file, 'rb'))

        
        try:
            if data.use_sup:
                sup_data = pickle.load(open(data.sup_file, 'rb'))
                sup_dataset.add_field('word', sup_data['word'][:data.sup_size])
                sup_dataset.add_field('raw_word', sup_data['word'][:data.sup_size])
                sup_dataset.add_field('gold_tree', sup_data['gold_tree'][:data.sup_size], padder=None,ignore_type=True)
                sup_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        except:
            data.use_sup = False

        train_dataset.add_field("word", train_data['word'])
        train_dataset.add_field("raw_word", train_data['word'])


        val_dataset.add_field("word", val_data['word'])
        val_dataset.add_field("raw_word", val_data['word'])
        test_dataset.add_field("word", test_data['word'])
        test_dataset.add_field("raw_word", test_data['word'])




        print('Training dataset size:', len(train_data['word']))


        # train_dataset.add_field("gold_tree", train_data['gold_tree'],padder=None,ignore_type=True)
        val_dataset.add_field("gold_tree", val_data['gold_tree'],padder=None,ignore_type=True)
        test_dataset.add_field("gold_tree", test_data['gold_tree'],padder=None,ignore_type=True)

        try:
            train_dataset.add_field("gold_tree", train_data['gold_tree'],padder=None,ignore_type=True)
        except:
            print("No gold tree4 training data")

        try:
            val_dataset.add_field('raw_tree', val_data['raw_tree'], padder=None, ignore_type=True)
            test_dataset.add_field('raw_tree', test_data['raw_tree'], padder=None, ignore_type=True)
        except:
            print("No raw tree found.")



        train_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        val_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        test_dataset.add_seq_len(field_name="word", new_field_name="seq_len")

        val_dataset = val_dataset.drop(lambda x:x['seq_len']==0, inplace=True)
        train_dataset = train_dataset.drop(lambda x:x['seq_len']==0, inplace=True)
        test_dataset = test_dataset.drop(lambda x: x['seq_len']==0, inplace=True)


        # if data.use_char:
        #     from .field import SubwordField
        #     char_field = SubwordField('char', pad="<PAD>", unk="<UNK>", subword_bos='<BOS>', subword_eos="<EOS>", fix_len=20)
        #     char_field.build(train_dataset['word'])
        #     train_dataset.apply_field(func=char_field.transform, field_name='word', new_field_name='char')
        #     val_dataset.apply_field(func=char_field.transform, field_name='word', new_field_name='char')
        #     test_dataset.apply_field(func=char_field.transform, field_name='word', new_field_name='char')
        #
        #     # train_dataset.set_ignore_type('char')
        #     # val_dataset.set_ignore_type('char')
        #     # test_dataset.set_ignore_type('char')
        #
        #     train_dataset.set_input('char')
        #     val_dataset.set_input('char')
        #     test_dataset.set_input('char')
        #     self.char_vocab_size = len(char_field.vocab)


        def clean_word(words):
            import re
            def clean_number(w):
                new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
                return new_w
            return [clean_number(word.lower()) for word in words]


        train_dataset.apply_field(clean_word, "word", "word")
        val_dataset.apply_field(clean_word, "word", "word")
        test_dataset.apply_field(clean_word, "word", "word")

        word_vocab.from_dataset(train_dataset, field_name="word")

        if data.use_sup:
            sup_dataset.apply_field(clean_word, "word", "word")
            word_vocab.index_dataset(sup_dataset, field_name="word")
            self.sup_dataset = sup_dataset
            self.sup_dataset.set_input("word", "seq_len", 'raw_word')
            self.sup_dataset.set_input('gold_tree')

        # if data.use_emb:
        #     emb = np.zeros((len(word_vocab), data.word_emb_size))
        #
        #     with open(data.emb_path, 'r', encoding="utf-8") as f:
        #         for line in f:
        #             line = line.split(" ")
        #             word = line[0]
        #             embedding = [float(l) for l in line[1:]]
        #             try:
        #                 word_idx = word_vocab.word2idx[word]
        #                 emb[word_idx] = embedding
        #             except:
        #                 continue
        #     self.emb = emb
        #
        # else:
        #     self.emb = None

        word_vocab.index_dataset(train_dataset, field_name="word")
        word_vocab.index_dataset(val_dataset, field_name="word")
        word_vocab.index_dataset(test_dataset, field_name="word")

        #drop length 1 sentences. As S->NT, while NT cannot generate single word in our
        #settings (only preterminals generate words
        self.val_dataset = val_dataset.drop(lambda x:x['seq_len']==1, inplace=True)
        self.train_dataset = train_dataset.drop(lambda x:x['seq_len']==1, inplace=True)
        self.test_dataset = test_dataset.drop(lambda x: x['seq_len']==1, inplace=True)

        self.word_vocab = word_vocab
        # self.char_vocab_size = len(char_field.vocab)
        self.train_dataset.set_input("word","seq_len",'raw_word')
        self.val_dataset.set_input("word","seq_len",'raw_word')
        self.test_dataset.set_input("word","seq_len",'raw_word')

        self.val_dataset.set_target('gold_tree')
        self.test_dataset.set_target("gold_tree",)

        try:
            self.train_dataset.set_input('gold_tree')
        except:
            pass

        try:
            self.val_dataset.set_input('raw_tree')
            self.test_dataset.set_input('raw_tree')
        except:
            pass


    def train_dataloader(self, max_len=60, min_len=2, epoch=1):
        args = self.hparams.train
        train_dataset = self.train_dataset.drop(lambda x:x['seq_len']>max_len, inplace=False)
        train_dataset = train_dataset.drop(lambda x:x['seq_len']<min_len, inplace=False)
        train_sampler = ByLengthSampler(dataset= train_dataset, batch_size=args.batch_size, shuffle=(epoch==1))
        return DataSetIter(dataset=train_dataset, batch_sampler= train_sampler)

    def val_dataloader(self, max_len=40):
        args = self.hparams.test
        self.val_dataset = self.val_dataset.drop(lambda x:x['seq_len']>max_len, inplace=False)
        if args.sampler == 'token':
            test_sampler = ConstantTokenNumSampler(seq_len=self.val_dataset.get_field("seq_len").content,
                                                max_token=args.max_tokens, num_bucket=args.bucket)
            return DataSetIter(self.val_dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                           pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None,
                           batch_sampler=test_sampler)
        elif args.sampler == 'batch':
            train_sampler = ByLengthSampler(dataset= self.val_dataset, batch_size=args.batch_size)
            return DataSetIter(dataset=self.val_dataset, batch_sampler= train_sampler)
        else:
            raise NotImplementedError



    def test_dataloader(self, max_len=40):
        args = self.hparams.test
        test_dataset = self.test_dataset.drop(lambda x:x['seq_len']>max_len, inplace=False)
        if args.sampler == 'token':
            test_sampler = ConstantTokenNumSampler(seq_len=test_dataset.get_field("seq_len").content,
                                                max_token=args.max_tokens, num_bucket=args.bucket)
            return DataSetIter(self.test_dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                           pin_memory=True, drop_last=False, timeout=0, worker_init_fn=None,
                           batch_sampler=test_sampler)
        elif args.sampler == 'batch':
            train_sampler = ByLengthSampler(dataset= test_dataset, batch_size=args.batch_size)
            return DataSetIter(dataset=test_dataset, batch_sampler= train_sampler)
        else:
            raise NotImplementedError


    def sup_dataloader(self, max_len=100):
        args = self.hparams.sup
        sup_dataset = self.sup_dataset.drop(lambda x:x['seq_len']>max_len, inplace=False)
        train_sampler = ByLengthSampler(dataset= sup_dataset, batch_size=args.batch_size)
        return DataSetIter(dataset=sup_dataset, batch_sampler= train_sampler)


'''
Same as (Kim et al, 2019)
'''
class ByLengthSampler(Sampler):
    def __init__(self, dataset, batch_size=4, shuffle=False):
        self.group = defaultdict(list)
        self.seq_lens = dataset['seq_len']
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size =  batch_size
        total = []
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        for idx, lst  in self.group.items():
            total = total + list(chunks(lst, self.batch_size))

        self.shuffle = shuffle
        self.total = total


    def __iter__(self):
        random.shuffle(self.total)
        for batch in self.total:
            yield batch


    def __len__(self):
        return len(self.total)






