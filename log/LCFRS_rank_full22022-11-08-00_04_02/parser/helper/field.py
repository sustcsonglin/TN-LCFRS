# -*- coding: utf-8 -*-

from collections import Counter

import torch



from collections import defaultdict
from collections.abc import Iterable


class Vocab(object):
    r"""
    Defines a vocabulary object that will be used to numericalize a field.
    Args:
        counter (~collections.Counter):
            :class:`~collections.Counter` object holding the frequencies of each value found in the data.
        min_freq (int):
            The minimum frequency needed to include a token in the vocabulary. Default: 1.
        specials (list[str]):
            The list of special tokens (e.g., pad, unk, bos and eos) that will be prepended to the vocabulary. Default: [].
        unk_index (int):
            The index of unk token. Default: 0.
    Attributes:
        itos:
            A list of token strings indexed by their numerical identifiers.
        stoi:
            A :class:`~collections.defaultdict` object mapping token strings to numerical identifiers.
    """

    def __init__(self, counter, min_freq=1, specials=[], unk_index=0):
        self.itos = list(specials)
        self.stoi = defaultdict(lambda: unk_index)

        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        self.extend([token for token, freq in counter.items()
                     if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], str):
            return [self.stoi[i] for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        return token in self.stoi

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)


    ### do not change
    def extend(self, tokens):
        try:
            self.stoi = defaultdict(lambda: self.unk_index)
            for word in self.itos:
                self.stoi[word]
        except:
            pass

        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
# Modified from Supar.

class RawField(object):
    r"""
    Defines a general datatype.

    A :class:`RawField` object does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequence):
        return self.preprocess(sequence)


class Field(RawField):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`Vocab` object. If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, use_vocab=True, tokenize=None, fn=None, min_freq=1):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn
        self.min_freq = min_freq
        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.tokenize is None:
            state['tokenize_args'] = None
        elif self.tokenize.__module__.startswith('transformers'):
            state['tokenize_args'] = (self.tokenize.__module__, self.tokenize.__self__.name_or_path)
            state['tokenize'] = None
        return state

    def __setstate__(self, state):
        tokenize_args = state.pop('tokenize_args', None)
        if tokenize_args is not None and tokenize_args[0].startswith('transformers'):
            from transformers import AutoTokenizer
            state['tokenize'] = AutoTokenizer.from_pretrained(tokenize_args[1]).tokenize
        self.__dict__.update(state)

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence):
        r"""
        Loads a single example using this field, tokenizing if necessary.
        The sequence will be first passed to ``dm_util`` if available.
        If ``tokenize`` is not None, the input will be tokenized.
        Then the input will be lowercased optionally.

        Args:
            sequence (list):
                The sequence to be preprocessed.

        Returns:
            A list of preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, sequences):

        r"""
        Constructs a :class:`Vocab` object for this field from the dataset.
        If the vocabulary has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A :class:`Dataset` object. One of the attributes should be named after the name of this field.
        """
        if hasattr(self, 'vocab') or not self.use_vocab:
            return
        # sequences = getattr(dataset, self.name)
        counter = Counter(token
                          for seq in sequences
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)


    def transform(self, sequence):
        sequence = self.preprocess(sequence)
        if self.use_vocab:
            sequence = self.vocab[sequence]
        if self.bos:
            sequence = [self.bos_index] + sequence
        if self.eos:
            sequence = sequence + [self.eos_index]
        return sequence


class SubwordField(Field):

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        self.subword_bos = kwargs.pop('subword_bos') if 'subword_bos' in kwargs else None
        self.subword_eos = kwargs.pop('subword_eos') if 'subword_eos' in kwargs else None
        if self.fix_len == -1:
            self.fix_len = 100000000
        super().__init__(*args, **kwargs)

        # for charlstm
        if self.subword_bos:
            self.specials.append(self.subword_bos)
        if self.subword_eos:
            self.specials.append(self.subword_eos)


    def build(self, sequences):
        if hasattr(self, 'vocab') or not self.use_vocab:
            return
        counter = Counter(piece
                          for seq in sequences
                          for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)


    def transform(self, seq):
        seq = [self.preprocess(token) for token in seq]

        if self.use_vocab:
            seq =  [  [self.vocab[i] if i in self.vocab else self.unk_index for i in token] if token else [self.unk_index]
                 for token in seq]

        if self.bos:
            seq = [[self.bos_index] ] + seq

        if self.eos:
            seq = seq + [[self.eos_index]]

        l = min(self.fix_len, max(len(ids) for ids in seq))
        seq = [ids[: l] for ids in seq]

        if self.subword_bos:
            seq =  [ [self.vocab[self.subword_bos]] +  s  for s in seq]

        if self.subword_eos:
            seq = [s + [self.vocab[self.subword_eos]] for s in seq]

        return seq
