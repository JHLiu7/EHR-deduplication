import logging
import pandas as pd 
import numpy as np
import pytorch_lightning as pl
import torch, os
import torch.nn as nn
from collections import defaultdict

from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from deduper import Simple_deduper, dedup_note_df
from utils import NOTE_CATEGORIES, _args4dedup, _task2target, _cohort2sets, _task2rawfile, _name4dedup

from transformers import LongformerTokenizerFast, RobertaTokenizerFast, BertTokenizerFast, AutoTokenizer

class DedupDataset(Dataset):
    def __init__(self, args, stay_df, deduper, raw_text_df=None, deduped_data=None, tokenizer=None, token2id=None, **kwargs):
        super().__init__()
        """Main torch dataset for modeling

            args: see options.py
            stay_df: cohort df 
            deduper: see deduper.py
            raw_data_df: if provided, load raw data
            deduped_data: if provided, load pre-deduped data
            tokenizer: for bert models
            token2id: for models using word embeddings

        """

        self.task = args.TASK

        self.deduper = deduper
        self.tokenizer = tokenizer 
        self.token2id = token2id

        self.dedup_args = _args4dedup(args)
        self.skip_dedup = args.skip_dedup
        self.skip_style = args.skip_style # head, tail, head-tail if skip dedup
        self.drop_type = args.drop_type # default no drop ''

        self.max_length = args.max_length
        self.max_note_length = args.max_note_length
        self.bert_max_length = args.bert_max_length

        self.max_sent_num = args.max_sent_num
        self.max_doc_num = args.max_doc_num

        self.silent = args.silent
        self.use_hierarch = True if 'hier' in args.MODEL_TYPE else False
        self.use_sent_hier= True if args.MODEL_TYPE == 'hier3' else False

        # load data with sent kept: (pt-doc-sent-token)
        self.sent_kept = True


        if not raw_text_df is None:
            assert deduped_data is None, "don't feed two data sources together"
            self.data, self.text_lens = self.load_data(stay_df, raw_text_df)
        else:
            assert raw_text_df is None, "don't feed two data sources together"
            self.data, self.text_lens = self.load_deduped_data(stay_df, deduped_data)


    def load_data(self, stay_df, raw_text_df):
        # dedup dataset vs cleaned dataset; concat string vs sep notes
        data, lens = [], []
        # self.nnotes = []
        for _, r in tqdm(stay_df.iterrows(), disable=self.silent, total=stay_df.shape[0]):

            hadm = r['HADM_ID']
            target = r[_task2target(self.task)]

            text_df = raw_text_df[raw_text_df.HADM_ID == hadm]

            notes = dedup_note_df(text_df, NOTE_CATEGORIES, self.deduper, **self.dedup_args)
            # note: list of tuples [(cat, text), ...] or [(cat, [sent, ...]), ...]

            parsed_note = self._parse_notes(notes) # a single string, or a seq of strings

            encoded_text, length = self._tokenize_text(parsed_note)

            data.append((encoded_text, target, hadm))
            lens.append(length)
            # self.nnotes.append(note)
        return data, lens

    def load_deduped_data(self, stay_df, hadm2deduped):
        data, lens = [], []
        for _, r in tqdm(stay_df.iterrows(), disable=self.silent, total=stay_df.shape[0]):

            hadm = r['HADM_ID']
            target = r[_task2target(self.task)]

            notes = hadm2deduped[hadm]

            parsed_note = self._parse_notes(notes) # a single string, or a seq of strings
            encoded_text, length = self._tokenize_text(parsed_note)

            data.append((encoded_text, target, hadm))
            lens.append(length)
        return data, lens


    def _tokenize_text(self, parsed_note):
        assert self.tokenizer != self.token2id, "need at least one to encode text; but don't feed them together, can't tell what to do"
        
        if not self.tokenizer is None:
            # assert type(parsed_note) != list, "don't support hierar with PTLM yet"
            if type(parsed_note) == list:
                parsed_note = ' '.join(parsed_note)
            else:
                assert type(parsed_note) == str
            encoded_text = self.tokenizer(parsed_note, max_length=self.bert_max_length, truncation=True, padding='max_length', return_token_type_ids=False)
            length = np.array(encoded_text.attention_mask).sum()
        
        else:
            # if type(parsed_note) == str:
            if not self.use_hierarch:
                # flat model, concat as a single string

                encoded_notes = [self._text2id(note, self.token2id) for note in parsed_note]
                if self.max_note_length > 0:
                    encoded_notes = [note[:self.max_note_length] for note in encoded_notes if len(note)>0]
                token_text = [i for j in encoded_notes for i in j]
                length = len(token_text[:self.max_length]) 

                # truncate or pad
                min_len = min(self.max_length, len(token_text))
                encoded_text = np.zeros(self.max_length)
                encoded_text[:min_len] = token_text[:min_len]

            # elif type(parsed_note) == list: # for hier modeling, full input
            else:
                if self.max_doc_num > 0:
                    parsed_note = parsed_note[:self.max_doc_num]
                    
                if (not self.sent_kept) or (not self.use_sent_hier):
                    # [text, text, ...], pt-doc-word
                    encoded_text = [self._text2id(note, self.token2id) for note in parsed_note]
                    encoded_text = [text[:self.max_note_length] for text in encoded_text if len(text)>0]
                    length = sum([len(t) for t in encoded_text])
                else:
                    # [[sent, sent,..], ...], pt-doc-sent-word
                    encoded_text, length = [], 0
                    for doc in parsed_note:
                        encoded_doc = [self._text2id(sent, self.token2id) for sent in doc]
                        encoded_text.append(encoded_doc[:self.max_sent_num])
                        length += sum([len(t) for t in encoded_doc])

        return encoded_text, length

    @staticmethod
    def _text2id(text, token2id):
        return [token2id[token.lower()] if token.lower() in token2id else token2id['<unk>'] for token in text.split()]

    def _parse_notes(self, notes):
        """
        notes: [(cat, text), ...]

        return: list of str or single str
        """
        if self.drop_type != '':
            notes = self._drop_cat(notes, self.drop_type)

        notes = [t for _,t in notes]

        if self.use_hierarch:
            ## return seqs
            if not self.use_sent_hier:
                notes = [' '.join(sents) for sents in notes]
            out = notes
        else:
            if self.sent_kept:
                notes = [' '.join(sents) for sents in notes]
            ## return single str
            if self.skip_dedup:
                if self.skip_style == 'tail':
                    out = notes[::-1]
                elif self.skip_style == 'headtail':
                    out = self._merge_select(notes, self.max_length)
                else:
                    out = notes
            else:
                out = notes
            # out = ' '.join(out)
        return out


    @staticmethod
    def _drop_cat(note_list, to_drop, categories=None):
        """
        note_list: [(cat, note), ...], ordered by chart time
        to_drop: radiology or radiology+nursing etc
        """
        if '+' in to_drop:
            drop = to_drop.split('+')
        else:
            drop = [to_drop]
        new_list = [(c, n) for c, n in note_list if c not in drop]
        if len(new_list) == 0:
            new_list = [(None, 'empty note')]
        return new_list


    @staticmethod
    def _merge_select(tlist, max_length):
        # skip style: headtail
        tlist = tlist.copy()
        tindex = list(range(len(tlist)))
        new_tlist, new_tindex, token_ct, i = [], [], 0, 0

        # select head/tail note one by one
        while token_ct < max_length and len(tlist) > 0:
            if i % 2 == 0:
                note = tlist.pop(0)
                idx = tindex.pop(0)
            else:
                note = tlist.pop(-1)
                idx = tindex.pop(-1)
            new_tlist.append(note)
            new_tindex.append(idx)
            token_ct += len(note.split())
            i += 1
        new_tlist, new_tindex = np.array(new_tlist, dtype=object), np.array(new_tindex)
        final_tlist = new_tlist[np.argsort(new_tindex)] # resume order
        return list(final_tlist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def _balance_df(df, fold=1):
    label = df.columns[-1]
    pos = df[df[label]==1]
    neg = df[df[label]==0]
    neg_ = neg.sample(n=len(pos)*fold, random_state=1233)
    bdf = pd.concat([pos, neg_]).sample(frac=1, random_state=1233) # shuffle
    return bdf

def _get_tokenizer(model_name):
    if model_name == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=f'PTM/{model_name}')
    elif model_name == 'clinical':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=f'PTM/{model_name}')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", cache_dir=f'PTM/{model_name}')
    elif model_name == 'longformer':
        tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096", cache_dir=f'PTM/{model_name}')
    else:
        raise NotImplementedError('lack of bert clf implementation')
    return tokenizer

def collate_bert(batch):
    x = defaultdict(list)
    y,stay = [],[]

    for xd, l, s in batch:
        x['input_ids'].append(xd['input_ids'])
        x['attention_mask'].append(xd['attention_mask'])

        y.append(l)
        stay.append(s)
    x = {k:torch.LongTensor(v) for k,v in x.items()}
    y = torch.tensor(y)
    return x, y, stay

def collate_w2v(batch):
    x = torch.tensor([b[0] for b in batch]).long() # already padded to max len
    y = torch.tensor( [b[1] for b in batch] )
    stay = np.array( [b[-1] for b in batch] )

    return x, y, stay

def collate_hier(batch):
    """
    batch: 
        x: [lists of various lengths]
        y: int/float

    return:
        x: (pad_note_tensor, note_lengths, stay_lengths, note_order)
        y: torch.tensor
    """

    note_series = [b[0] for b in batch]
    notes = [torch.LongTensor(note) for series in note_series for note in series]
    note_lengths = torch.tensor([len(note) for note in notes])
    stay_lengths = torch.tensor([len(series) for series in note_series])
    stay_order = _create_stay_order(stay_lengths)

    pad = pad_sequence(notes, batch_first=True)
    x_tuple = (pad, note_lengths, stay_lengths, stay_order)

    y = torch.tensor( [b[1] for b in batch] )
    stay = np.array( [b[-1] for b in batch] )
    
    return x_tuple, y, stay

def collate_hier3(batch):
    """collate a batch to enable sentence-level hierarchical modeling

    """
    note_series = [b[0] for b in batch]

    notes = [note for stay in note_series for note in stay]
    sents = [torch.LongTensor(sent) for note in notes for sent in note]

    sent_lengths = torch.tensor([len(sent) for sent in sents])
    note_lengths = torch.tensor([len(note) for note in notes])
    stay_lengths = torch.tensor([len(series) for series in note_series])

    note_order = _create_stay_order(stay_lengths)
    sent_order = _create_stay_order(note_lengths)

    assert len(sent_lengths) == sum(note_lengths)

    pad = pad_sequence(sents, batch_first=True)
    x_tuple = (pad, sent_lengths, note_lengths, stay_lengths, note_order, sent_order)

    y = torch.tensor( [b[1] for b in batch] )
    stay = np.array( [b[-1] for b in batch] )
    return x_tuple, y, stay

def _create_stay_order(stay_lengths):
    bs, ml = len(stay_lengths), max(stay_lengths)

    stay_msk = torch.zeros((bs, ml)).long()
    stay_order = torch.zeros((bs, ml)).long()
    for i, s in enumerate(stay_lengths):
        stay_msk[i, :s] = 1

    stay_order[stay_msk.bool()] = torch.arange(1,stay_lengths.sum()+1)
    return stay_order


class TextDedupDataModule(pl.LightningDataModule):
    """pytorch lightning data module
    """
    def __init__(self, args):
        super().__init__()

        # load cohort info
        task = args.TASK
        cohort_df = pd.read_csv(os.path.join(args.COHORT_DIR, f'new-{task}.csv'))
        if args.debug:
            cohort_df = cohort_df.head(1000)
        # split train/val/test sets
        self.tr_df, self.val_df, self.te_df = _cohort2sets(cohort_df, task)

        if args.balance_df > 0:
            n = args.balance_df
            self.tr_df, self.val_df, self.te_df = [_balance_df(df, fold=n) for df in (self.tr_df, self.val_df, self.te_df)]

        # load raw text data or deduped data
        dedup_args = _args4dedup(args)
        deduped_data_path = os.path.join(args.TEXT_DEDUP_DIR, _name4dedup(task, dedup_args))
        if os.path.isfile(deduped_data_path):
            self.deduped_data = pd.read_pickle(deduped_data_path)
            self.raw_text_df = None
            print('Loaded pre-processed data from', deduped_data_path)
        else:
            raw_data_path = os.path.join(args.TEXT_RAW_DIR, _task2rawfile(task)) #, args.debug))
            self.raw_text_df = pd.read_pickle(raw_data_path)
            self.deduped_data = None

        # get deduper
        self.deduper = Simple_deduper(remove_phi=True, remove_num_unit=True, remove_punkt=True)

        # get tokenizer or token2id
        if args.MODEL_TYPE == 'bert': 
            self.tokenizer = _get_tokenizer(args.bert_model_name)
            self.token2id = None
        else:
            self.tokenizer = None
            self.token2id = pd.read_pickle(os.path.join(args.TEXT_RAW_DIR, 'token2id.dict'))

        # setup options
        self.args = args 
        self.batch_size = args.batch_size

        # collator: bert vs w2v
        collate_dict = {
            'w2v': collate_w2v,
            'bert': collate_bert,
            'hier': collate_hier,
            'hier3': collate_hier3
        }
        self.my_collate = collate_dict[args.MODEL_TYPE]

    def _init_dataset(self, df):
        return DedupDataset(self.args, df, deduper=self.deduper, 
                raw_text_df=self.raw_text_df, deduped_data=self.deduped_data, 
                tokenizer=self.tokenizer, token2id=self.token2id)

    def setup(self, stage=None):
        if stage == None:
            self.train_dataset, self.val_dataset, self.test_dataset = [self._init_dataset(df) for df in (self.tr_df, self.val_df, self.te_df)]
            print(f'\nActual train size: {len(self.train_dataset)}, val size: {len(self.val_dataset)}, test size: {len(self.test_dataset)}')

            # if self.count_note_length:
            print('\nAvg input text lengths\nTrain\tValidation\tTest')
            tr, val, te = [dset.text_lens for dset in (self.train_dataset, self.val_dataset, self.test_dataset)]
            print(f'{np.mean(tr):.1f}({np.std(tr):.1f})\t{np.mean(val):.1f}({np.std(val):.1f})\t{np.mean(te):.1f}({np.std(te):.1f})\n')
        
        elif stage == 'test':
            self.test_dataset = self._init_dataset(self.te_df) 
            print(f'Actual test size: {len(self.test_dataset)}')
            print('Avg input text lengths')
            te = self.test_dataset.text_lens 
            print(f'{np.mean(te):.1f}({np.std(te):.1f})\n')
        
    def train_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.my_collate)
    def val_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate)
    def test_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate)


