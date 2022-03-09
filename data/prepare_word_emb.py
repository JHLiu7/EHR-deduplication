import pandas as pd
import numpy as np
import pickle as pk
import os, sys, re 
from datetime import datetime

from shutil import copyfile
from tqdm import tqdm
import argparse
from collections import Counter
from gensim.models import KeyedVectors
from nltk import word_tokenize


def get_embeddings(words, pretrained_embed_dir):
    print("loading biowordvec...")
    model = KeyedVectors.load_word2vec_format(os.path.join(pretrained_embed_dir, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin'), binary=True)
    print("loaded, start to get embed for tokens")

    model_vocab = set(model.index2word)

    # valid word in pretrained embeddings
    valid_word = []
    oov = []
    for w in words:
        if w in model_vocab:
            valid_word.append(w)
        else:
            oov.append(w)
    # print(f'{len(oov)} oov:', oov) # may contain lines not properly splitted; can fix it by using preprocessing script of biowordvec

    # vocab dict
    token2id = {}
    token2id['<pad>'] = 0 
    for word in valid_word:
        token2id[word] = len(token2id)
    token2id['<unk>'] = len(token2id)

    # embed; pad zero, unk random
    dim = model.vectors.shape[1]
    embeddings = np.zeros( (len(valid_word)+2, dim), dtype=np.float32)
    embeddings[0] = np.zeros(dim, )
    embeddings[-1] = np.random.randn(dim, )
    for i, w in enumerate(valid_word):
        embeddings[i+1] = model[w]
    print("embed shape", embeddings.shape)

    return token2id, embeddings

def _strip_phi(t):
    t = re.sub(r'\[\*\*.*?\*\*\]', ' ', t)
    t = re.sub(r'_', ' ', t)
    t = re.sub(r"`", '', t)
    t = re.sub(r"''", '', t)
    t = re.sub(r'"', '', t)
    return t

def _tokenize(text):
    text = _strip_phi(text)
    return [w.lower() for w in word_tokenize(text)]

def get_freq_words(all_text_df, min_freq):
    all_text_df = all_text_df.drop_duplicates(subset=['TEXT'])
    print(f'processing {len(all_text_df)} unique notes')

    tqdm.pandas()
    print('clean text and tokenize')
    text_tokens = all_text_df['TEXT'].progress_apply(_tokenize)

    # all_tokens = np.concatenate(text_tokens.tolist())
    # print(f'found {len(all_tokens)} tokens in all notes')
    # token_count = Counter(all_tokens)
    token_count = {}
    for text in tqdm(text_tokens.tolist()):
        for token in text:
            if token not in token_count:
                token_count[token] = 0
            token_count[token] += 1
    print(f'found {len(token_count)} unique tokens')

    # token_freq = [w for (w,c) in token_count.most_common() if c >= min_freq]  
    token_freq = [w for w,c in token_count.items() if c >= min_freq]  
    print(f'{len(token_freq)} freq tokens with min freq of {min_freq}')

    return token_freq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--OUT_DIR", default='text_cohorts_raw', type=str)

    parser.add_argument("--pretrained_embed_dir", default='', type=str)
    parser.add_argument("--min_freq", default=5, type=int)

    parser.add_argument("--debug", "-d", action="store_const", const=True, default=False)    
    args = parser.parse_args()

    pfiles = ['mort_1daywindow_33515hadms_305737notes.p', 'readm_fullwindow_34028hadms_854570notes.p', 'drglos_2daywindow_37303hadms_470727notes.p']

    if args.debug:
        all_text_df = pd.read_pickle(os.path.join(args.OUT_DIR, pfiles[0])).head(100)
    else:
        print('loading all raw notes')
        all_text_df = pd.concat([pd.read_pickle(os.path.join(args.OUT_DIR, p)) for p in pfiles])
    
    print('counting words..')
    words = get_freq_words(all_text_df, args.min_freq)

    token2id, embedding = get_embeddings(words, args.pretrained_embed_dir)

    print('dumping t2i and embed')
    with open(os.path.join(args.OUT_DIR, 'token2id.dict'), 'wb') as outf:
        pk.dump(token2id, outf)
    np.save(os.path.join(args.OUT_DIR, 'embedding.npy'), embedding)

