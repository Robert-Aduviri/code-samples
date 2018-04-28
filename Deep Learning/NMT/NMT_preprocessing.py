import unicodedata
import re
import random
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torchtext import data

PAD_token = 0
SOS_token = 1 # Start Of Sentence
EOS_token = 2 # End of Sentece

############## READ DATA AND FILTER SENTENCES ##########################

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'SOS', 2:'EOS'}
        self.n_words = 3 # Counting SOS and EOS
        
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)    

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s) # separate .!? from words
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s) # delete other characters
    return ' '.join(s.split())

def read_langs(path, lang1, lang2, reverse=False):
    print('Reading lines...')
    
    # Read lines
    
    with open(path, encoding="utf-8") as f:
        lines = f.read().strip().split('\n')
    
    # Split each line into [lang1, lang2] normalized pairs
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # Reverse pairs and build lang dictionaries
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

# for english only
filtered_prefixes = (
    'i am ', 'i m ',
    'he is ', 'he s ',
    'she is ', 'she s ',
    'you are ', 'you re '
)

def filter_pair(p, max_length):
    # less than MAX_LENGTH words
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length # and \
           # p[0].startswith(filtered_prefixes) # if english is the second element
        
def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

def prepare_data(path, lang1, lang2, max_length=10, reverse=False):
    input_lang, output_lang, pairs = read_langs(path, lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')
    
    pairs = filter_pairs(pairs, max_length)
    print(f'Trimmed to {len(pairs)} sentence pairs')
    
    print('Indexing words...')
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    return input_lang, output_lang, np.array(pairs)

############## GENERATE VECTORS ##########################

def construct_vectors(pairs, vector_name_in='fasttext.en.300d', vector_name_out='fasttext.es.300d'):
    lang_in = pd.DataFrame(pairs[:, 0], columns=["lang_in"])
    lang_out = pd.DataFrame(pairs[:, 1], columns=["lang_out"])

    lang_in.to_csv('lang_in.csv', index=False)
    lang_out.to_csv('lang_out.csv', index=False)

    lang_in = data.Field(sequential=True, lower=True)
    lang_out = data.Field(sequential=True, lower=True)

    mt_lang_in = data.TabularDataset(
        path='lang_in.csv', format='csv',
        fields=[('lang_in', lang_in)])
    mt_lang_out = data.TabularDataset(
        path='lang_out.csv', format='csv',
        fields=[('lang_out', lang_out)])

    lang_in.build_vocab(mt_lang_in)
    lang_out.build_vocab(mt_lang_out)

    lang_in.vocab.load_vectors(vector_name_in)
    lang_out.vocab.load_vectors(vector_name_out)
    
    return lang_in, lang_out

############## GENERATE BATCHES ##########################

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def random_batch(batch_size, lang_source, lang_target, pairs, USE_CUDA=False):
    input_seqs = []
    target_seqs = []

    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(lang_source, pair[0]))
        target_seqs.append(indexes_from_sentence(lang_target, pair[1]))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        
    return input_var, input_lengths, target_var, target_lengths

def variable_from_sentence(lang, sentence, USE_CUDA=False):
    indexes = indexes_from_sentence(lang, sentence)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair, input_lang, output_lang, USE_CUDA=False):
    input_variable = variable_from_sentence(input_lang, pair[0], USE_CUDA)
    target_variable = variable_from_sentence(output_lang, pair[1], USE_CUDA)
    return (input_variable, target_variable)