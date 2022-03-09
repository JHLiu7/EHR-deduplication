from multiprocessing import Pool
import pandas as pd 
import numpy as np 
import pickle as pk
import string, re, os
import argparse, logging
from nltk import word_tokenize, ngrams
from tqdm import tqdm

from collections import OrderedDict, defaultdict
from difflib import Differ

from utils import NOTE_CATEGORIES, _strip_phi, _jaccard_dist, _is_num_unit, _fix_category, _args4dedup, _name4dedup

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')



class Simple_deduper:
    def __init__(self, remove_phi=True, remove_num_unit=False, remove_punkt=True):
        """Clean and dedup one note series

        Args:
            remove_phi (bool, optional): Remove de-identified placeholders in text. Defaults to True.
            remove_num_unit (bool, optional): Remove words consist of only numbers. Defaults to False.
            remove_punkt (bool, optional): Remove words that are punctuations. Defaults to True.
        """
        self.remove_phi = remove_phi
        self.remove_num_unit = remove_num_unit
        self.remove_punkt = remove_punkt

    def __call__(self, text_series, method='line set', return_idx=False, *args, **kwargs):
        """Call deduper

        Args:
            text_series (List): List of notes from a single case
            method (str, optional): Dedup methods: line set, note. Defaults to 'line set'.
            return_idx (bool, optional): Return idx of selected notes. Defaults to False.

        Raises:
            NotImplementedError: Dedup method not implemented

        Returns:
            List: List of deduped ntoes
        """
        # pre process
        text_series = self.check_text_series(text_series)

        if method == 'line set':
            new_text_series, kept_idx = self._dedup_by_line(text_series, 'line set')
        elif method == 'jnote':
            new_text_series, kept_idx = self._dedup_by_note(text_series, **kwargs)
        elif method == 'seq match':
            self.d = Differ()
            new_text_series, kept_idx = self._dedup_by_line(text_series, 'seq match')
        elif method == 'jaccard':
            # archived
            new_text_series, kept_idx = self._dedup_with_word(text_series, **kwargs)
        else:
            raise NotImplementedError

        # post process
        new_text_series = self._postprocess(new_text_series)

        if return_idx:
            return new_text_series, kept_idx
        else:
            return new_text_series

    def simply_clean(self, text_series):
        """
        apply same pre/post processing procedures but w/o dedup
        """
        text_series = self.check_text_series(text_series)
        return self._postprocess(text_series)

            
    def check_text_series(self, text_series):
        # assert len(text_series) > 1, 'No need to dedup, only one text'
        assert type(text_series[0]) == str, 'Each piece of text should be in string and contains \n'
        if self.remove_phi:
            text_series = [_strip_phi(text) for text in text_series]
        return text_series
    
    def _postprocess(self, new_text_series):
        """
        new_text_series: list of text strings 
        return: list of text strings
        """

        # drop emtpy
        new_text_series = [text for text in new_text_series if len(text.strip())>0]

        # split sent
        text_series_sent = [text.split('.') for text in new_text_series]

        # clean sent 
        text_series_sent = [[sent.strip(string.punctuation+' ') + '.' for sent in doc] for doc in text_series_sent]

        # handle num/punkt
        text_series = [] # (pt - doc - sent)
        for doc in text_series_sent:
            doc_series = []
            for sent in doc:
                tokens = word_tokenize(sent)
                if self.remove_num_unit:
                    tokens = [t for t in tokens if not _is_num_unit(t)]
                if self.remove_punkt:
                    tokens = [t for t in tokens if not t in string.punctuation]
                if len(tokens) > 0:
                    sent = ' '.join(tokens)
                    doc_series.append(sent)
            text_series.append(doc_series)

        return text_series

        
    # def dedup_by_jaccard_thresh(self, text_series, thresh=0.5, global_thresh=0.2, n_gram=0):
    #     # archived
    #     new_text_series, kept_idx = self._dedup_with_word(text_series, 'jaccard', thresh, global_thresh, n_gram)
    #     return new_text_series, kept_idx

    # def dedup_by_note(self, text_series, thresh=0.5, n_gram=1):
    #     # to replace dedup w jaccard
    #     new_text_series, kept_idx = self._dedup_note(text_series, threshold=thresh, n_gram=n_gram)
    #     return new_text_series, kept_idx

    def _dedup_by_note(self, text_series, thresh=.55, n_gram=1, **kwargs):
        
        ref_text_series = text_series.copy()

        if len(text_series) == 1:
            return text_series, np.array([0])

        if n_gram <= 1:
            text_series = [text.split() for text in text_series]
        else:
            text_series = [ ['-'.join(g) for g in ngrams(text.split(), n_gram)] for text in text_series]
        
        # compare with all prior notes; if similar to any prior one, drop note
        notes_kept = text_series[:1]
        idx_kept = [0]

        for i, note in enumerate(text_series[1:]):
            if len(note) == 0:
                continue

            # jaccard distance
            scores = [_jaccard_dist(note, prior_note) for prior_note in notes_kept]

            # higher dist means dissimilar note
            # if note is dissimilar with all prior notes, keep it
            if min(scores) > thresh:
                idx_kept.append(i+1)
                notes_kept.append(note)


        # select notes
        idx_kept = np.array(idx_kept)
        text_series = np.array(ref_text_series, dtype=object)
        new_text_series = list(text_series[idx_kept])

        return new_text_series, idx_kept

         
    def _dedup_by_line(self, text_series, method='line set'):
        if len(text_series) == 1:
            return text_series, np.array([0])
        text_series = [text.splitlines() for text in text_series]
        new_text_series = []
   
        # get base text for comparison
        state_text = [t.strip() for t in text_series[0]]
        new_text_series.append(state_text)
                
        for text in text_series[1:]:
            # prepare state text
            state_text = list(OrderedDict.fromkeys(state_text))

            # get next text
            text = [t.strip() for t in text]
            next_text = list(OrderedDict.fromkeys(text)) # dedup

            # compare
            if method == 'line set':
                new_piece = [l for l in next_text if l not in state_text]
            elif method == 'seq match':
                res = list(self.d.compare(state_text, next_text))
                new_piece = [l.strip('+ ') for l in res if l.startswith('+')]
                
            # append and update
            new_text_series.append(new_piece)
            state_text.extend(new_piece)

        final_text_series, to_keep = [], []
        for i, text in enumerate(new_text_series):
            if len(text) > 0:
                text_ = ' '.join(text).strip()
                if len(text_) > 0:
                    # non-empty note
                    final_text_series.append(text_)
                    to_keep.append(i)
                else:
                    continue
            else:
                continue
        to_keep = np.array(to_keep)
        return final_text_series, to_keep


    def _dedup_with_word(self, text_series, method='jaccard', thresh=0.5, global_thresh=0.2, n_gram=0):
        ### archived
        ref_text_series = text_series.copy()

        if len(text_series) == 1:
            return text_series, np.array([0])

        if n_gram <= 1:
            text_series = [text.split() for text in text_series]
        else:
            text_series = [ ['-'.join(g) for g in ngrams(text.split(), n_gram)] for text in text_series]
        
        # if token-based sim over a threshold, discard the text
        state_tokens = text_series[0].copy()
        full_tokens = state_tokens.copy()
        
        to_keep = [0]
        for i, tokens in enumerate(text_series[1:]):
            if len(tokens) == 0:
                continue
            dist_last = _jaccard_dist(state_tokens, tokens)
            dist_full = _jaccard_dist(full_tokens, tokens)
            
            if (dist_last > thresh) and (dist_full > global_thresh):
                to_keep.append(i+1)
                
            state_tokens = tokens
            full_tokens.extend(tokens)
        
        # select notes
        to_keep = np.array(to_keep)
        text_series = np.array(ref_text_series, dtype=object)
        new_text_series = list(text_series[to_keep])
        
        return new_text_series, to_keep
      

def dedup_note_df(df, categories, deduper, method='line set', skip_dedup=False, ignore_type=False, **kwargs):
    """
    skip_dedup: simply clean text, no de-duplication performed
    ignore_type: de-dup on whole notes; otherwise de-dup on sub-series of each note category

    return:
        if ignore type: list of separate notes 
        else: list of separate note tuple (cat, note) (deduped based on each cat) in temporal order
    """

    df = df.copy().reset_index()
    df['category'] = df['CATEGORY'].apply(_fix_category)

    if skip_dedup:
        cats = df['category'].tolist()
        cleaned = deduper.simply_clean(df['TEXT'].tolist())
        out_text = [(c, t) for c, t in zip(cats, cleaned)]
    else:
        if ignore_type:
            # cats = df['category'].tolist()
            deduped, kept_idx = deduper(df['TEXT'].tolist(), method, return_idx=True, **kwargs)

            orig_idx = df.index.values
            to_keep = np.zeros(len(orig_idx), dtype=np.int8)
            to_keep[kept_idx] = 1
            new_idx = orig_idx[to_keep.astype(bool)]

            cats = df['category'][new_idx]

            out_text = [(c, t) for c, t in zip(cats, deduped)]
        else:
            # dedup within note type, return notes in original temp order
            cat_notes, cat_idx = [], []
            for cat in categories:
                notes_cat = df[df['category']==cat]['TEXT'].tolist()
                orig_idx = df[df['category']==cat].index.values

                if len(notes_cat) == 0:
                    continue

                deduped, kept_idx = deduper(notes_cat, method, return_idx=True, **kwargs)

                to_keep = np.zeros(len(orig_idx), dtype=np.int8)
                to_keep[kept_idx] = 1
                new_idx = orig_idx[to_keep.astype(bool)]

                cat_notes.extend([(cat, t) for t in deduped])
                cat_idx.extend(new_idx)

            tmp_notes = np.array(cat_notes, dtype=object)
            orig_order = np.argsort(np.array(cat_idx))
            out_text = tmp_notes[orig_order].tolist()

    return out_text # [(c,t)...]

def _count_length(text_series, ignore_type=False, categories=None):
    if ignore_type:
        return len(' '.join([t for c,t in text_series]).split())
    else:
        # categories - [(cat_title, note), ...]
        assert categories != None
        lens = []
        for cat in categories:
            notes = [n for c,n in text_series if c == cat]
            text = ' '.join(notes)
            lens.append(len(text.split()))
        return np.array(lens)

def _calc_len_agg(lens, note_categories, ignore_type=False):
    if ignore_type:
        # all notes
        nmean, nstd = np.mean(lens), np.std(lens)
        line = f'{nmean:.1f}({nstd:.1f})'
    else:
        a = np.array(lens)
        stat = [o for tup in 
            [(i,j) for i,j in zip(a.mean(0), a.std(0))]
            for o in tup]
        # print('\t'.join(note_categories))
        line = '{:.1f}({:.1f})\t{:.1f}({:.1f})\t{:.1f}({:.1f})\t{:.1f}({:.1f})'.format(*stat)
    return line


def process_cohort_df_with_all_dedup(cohort_df, all_text_df, list_dedup_args, task, n_jobs=1):
    """
    list dedup args: the number of dedups to apply
    
    """

    logging.info(f'Processing/deduplicating text series for {task.upper()} with {len(list_dedup_args)} methods')

    deduper = Simple_deduper(remove_phi=True, remove_num_unit=True, remove_punkt=True)
    data_dicts, counts = [], []

    for i, dedup_args in enumerate(list_dedup_args):
        hadm2text, lens = {}, []
        
        if n_jobs <= 1:
            for _, r in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df)):
                hadm = r['HADM_ID']
                text_df = all_text_df[all_text_df.HADM_ID==hadm]

                text_series = dedup_note_df(text_df, NOTE_CATEGORIES, deduper, **dedup_args)
                hadm2text[hadm] = text_series # (pt-doc-sent)

                text_series = [(c, ' '.join(doc))  for c, doc in text_series] # (pt-doc)
                lens.append(_count_length(text_series, dedup_args['ignore_type'], NOTE_CATEGORIES))
        else:
            engine = DedupEngine(all_text_df, deduper, dedup_args)
            hadms = cohort_df.HADM_ID.tolist()
            logging.info(f'processing {len(hadms)} cases with {n_jobs} processes')
            with Pool(n_jobs) as pool:
                results = pool.map(engine, hadms)
            
            for hadm, (text_series, text_len) in zip(hadms, results):
                hadm2text[hadm] = text_series
                lens.append(text_len)

        data_dicts.append(hadm2text)
        counts.append(_calc_len_agg(lens, NOTE_CATEGORIES, ignore_type=dedup_args['ignore_type']))
        logging.info(f'{i+1}/{len(list_dedup_args)} processed')
        
    print(f'Finished for {task.upper()}')
    if not dedup_args['ignore_type']:
        print('\t'.join(NOTE_CATEGORIES))
    print('\n'.join(counts))

    return data_dicts


class DedupEngine(object):
    """To enable multiprocess running
    """
    def __init__(self, all_text_df, deduper, dedup_args) -> None:
        super().__init__()
        self.all_text_df = all_text_df
        self.deduper = deduper 
        self.dedup_args = dedup_args
    def __call__(self, hadm):
        text_df = self.all_text_df[self.all_text_df.HADM_ID==hadm]
        text_series = dedup_note_df(text_df, NOTE_CATEGORIES, self.deduper, **self.dedup_args)
        text_series_ = [(c, ' '.join(doc))  for c, doc in text_series] # (pt-doc)
        text_len = _count_length(text_series_, self.dedup_args['ignore_type'], NOTE_CATEGORIES)
        return text_series, text_len


if __name__ == '__main__':
    """Running the script will perform deduplication on all four cohorts and save temp data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--COHORT_DIR", default='data/cohorts', type=str)
    parser.add_argument("--TEXT_RAW_DIR", default='data/text_cohorts_raw', type=str)
    parser.add_argument("--TEXT_DEDUP_DIR", default='data/text_cohorts_dedup', type=str)

    parser.add_argument('--consider_types', action="store_const", const=True, default=False, help='If set true, only compare note with prior notes of the same type during deduplication.')
    parser.add_argument('--jthresh', type=float, default=0.55)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument("--debug", "-d", action="store_const", const=True, default=False)
    args = parser.parse_args()

    os.makedirs(args.TEXT_DEDUP_DIR, exist_ok=True)


    tasks, pnames = ['mort', 'readm', 'drg', 'los'], ['mort', 'readm', 'drglos']
    task2p = {
        'mort': 'mort', 'readm': 'readm', 'drg': 'drglos', 'los': 'drglos'
    }
    p2files = {
        'mort': 'mort_1daywindow_33515hadms_305737notes.p',
        'readm': 'readm_fullwindow_34028hadms_854570notes.p',
        'drglos': 'drglos_2daywindow_37303hadms_470727notes.p',
    }

    
    # dedup candidates
    dedup_args_raw = {
        'method': None, 'ignore_type': True, 'skip_dedup': True
    }
    dedup_args_line= {
        'method': 'line set', 'ignore_type': True, 'skip_dedup': False
    }
    dedup_args_jacc= {
        'method': 'jaccard', 'ignore_type': True, 'skip_dedup': False,
        'thresh': 0.5, 'global_thresh': 0.2, 'n_gram': 1
    }
    dedup_args_note= {
        'method': 'jnote', 'ignore_type': True, 'skip_dedup': False,
        'thresh': args.jthresh, 'n_gram': 1
    }
    list_dedup_args = [dedup_args_raw, dedup_args_line, dedup_args_note]


    # if consider type
    list_dedup_args_type = []
    for dedup_args in list_dedup_args:
        new_args = dedup_args.copy()
        new_args['ignore_type'] = False
        list_dedup_args_type.append(new_args)


    for task in tasks:
        cohort_df = pd.read_csv(os.path.join(args.COHORT_DIR, f'new-{task}.csv'))
        all_text_df = pd.read_pickle(os.path.join(args.TEXT_RAW_DIR, p2files[task2p[task]]))

        if args.debug:
            cohort_df = cohort_df.head(1000)
            task = 'sample_'+task

        logging.info('Ignore Note Types')
        list_hadm2text = process_cohort_df_with_all_dedup(cohort_df, all_text_df, list_dedup_args, task, args.n_jobs)

        for dedup_args, hadm2text in zip(list_dedup_args, list_hadm2text):
            fname = _name4dedup(task, dedup_args)
            with open(os.path.join(args.TEXT_DEDUP_DIR, fname), 'wb') as outf:
                pk.dump(hadm2text, outf)
            logging.info(f'Processed type-ignored file dumped to {fname}')


        if args.consider_types:
            logging.info('Consider Note Types')
            list_hadm2text_type = process_cohort_df_with_all_dedup(cohort_df, all_text_df, list_dedup_args_type, task, args.n_jobs)
            # discard raw, which is the same as ignore type 
            for dedup_args, hadm2text in zip(list_dedup_args_type[1:], list_hadm2text_type[1:]):
                fname = _name4dedup(task, dedup_args)
                with open(os.path.join(args.TEXT_DEDUP_DIR, fname), 'wb') as outf:
                    pk.dump(hadm2text, outf)
                logging.info(f'Processed type-aware file dumped to {fname}')


