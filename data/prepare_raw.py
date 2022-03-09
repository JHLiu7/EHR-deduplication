import re, string
import pandas as pd
import numpy as np
import pickle as pk
import os, sys 
from datetime import datetime
from tqdm import tqdm
import argparse
from collections import Counter, OrderedDict

def get_raw_note_df(MIMIC_DIR, cohort_df, input_time_window='1 day'):
    """
    MIMIC_DIR: NOTEEVENTS.csv.gz
    cohort_df: hadm, intime
    input_time_window: anchored to icu adm
    """

    hadms = cohort_df['HADM_ID'].astype(int).tolist()
    print(f'Processing {len(hadms)} hadms..')

    # 1. load noteevnts
    col_keep = ['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'DESCRIPTION', 'TEXT']
    note = pd.read_csv(os.path.join(MIMIC_DIR, 'NOTEEVENTS.csv.gz'), usecols=col_keep, dtype={'CHARTTIME':object})

    note.dropna(subset=['HADM_ID'], inplace=True)
    note['HADM_ID'] = note['HADM_ID'].astype(int)

    note = note[note['HADM_ID'].isin(hadms)]
    print(f'Found {len(note)} raw notes for the cohort')

    # 2. remove discharge summary and empty/same notes
    note = note[note.CATEGORY != 'Discharge summary']
    print('..after removing d-summary count', len(note))

    def _strip_len(x): return len(x.strip())
    note['length'] = note['TEXT'].apply(_strip_len)
    note = note[note['length']>0]
    print('..after removing empty (zero-len) notes', len(note))

    note = note.drop_duplicates()
    print('..after removing exact same notes', len(note))

    # 3. fix charttime
    note['CHARTTIME_fix'] = note.apply(lambda r: r['CHARTTIME'] if type(r['CHARTTIME']) is str else r['CHARTDATE'], axis=1)
    note['CHART_TIME'] = pd.to_datetime(note['CHARTTIME_fix'])

    # 4. filter by time and input window
    if input_time_window is None:
        col_drop = ['CHARTDATE', 'CHARTTIME', 'CHARTTIME_fix', 'length']
        print(f'{len(note)} notes w/o filtering on time')
    else:
        note = note.merge(cohort_df[['HADM_ID', 'INTIME']], on=['HADM_ID'])
        time_mask = note['CHART_TIME'] <= pd.to_datetime(note['INTIME']) + pd.Timedelta(input_time_window)
        note = note[time_mask]
        print(f'{len(note)} notes filtered by input time threshold')
        col_drop = ['CHARTDATE', 'CHARTTIME', 'INTIME', 'CHARTTIME_fix', 'length']

    # 5. sort
    note = note.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHART_TIME', 'CATEGORY', 'length'])
    
    ncol = note.columns.drop(col_drop)
    text_df = note[ncol]
    return text_df

def prepare_for_cohort(cohort_df, out_dir, cohort_name, input_window, MIMIC_DIR):
    print('*'*30, '\nPreparing', cohort_name)
    text_df = get_raw_note_df(MIMIC_DIR=MIMIC_DIR, cohort_df=cohort_df, input_time_window=input_window)
    hadms_with_text = text_df.HADM_ID.unique().tolist()

    hadms = len(hadms_with_text)
    ncount = len(text_df)
    window = input_window.replace(' ', '') if input_window else 'full'
    
    outname = f'{cohort_name}_{window}window_{hadms}hadms_{ncount}notes.p'
    outfile = os.path.join(out_dir, outname)
    with open(outfile, 'wb') as outf:
        pk.dump(text_df, outf)

    print(outname, 'saved to', out_dir)

    return hadms_with_text



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--COHORT_DIR", default='cohorts', type=str)
    parser.add_argument("--MIMIC_DIR", default='', type=str)
    parser.add_argument("--OUT_DIR", default='text_cohorts_raw', type=str)

    parser.add_argument("--debug", "-d", action="store_const", const=True, default=False)
    args = parser.parse_args()

    os.makedirs(args.OUT_DIR, exist_ok=True)

    tasks = ['mort', 'readm', 'drg', 'los']
    mort, readm, drg, los = [pd.read_csv(os.path.join(args.COHORT_DIR, f'cohort-{task}.csv')) for task in tasks]
    if args.debug:
        mort, readm, los = [df.head(40) for df in (mort, readm, los)]
    task2df = {
        'mort': mort, 'readm': readm, 'drg': drg, 'los': los
    }

    # hadms in drg are included in los, so we only need to save three data files
    assert np.in1d(drg.HADM_ID.values, los.HADM_ID.values).all()
    task2p = {
        'mort': 'mort', 'readm': 'readm', 'drg': 'drglos', 'los': 'drglos'
    }
    task2w = {
        'mort': '1 day', 'readm': None, 'drg': '2 day', 'los': '2 day'
    }

    for task in ['mort', 'readm']:
        hadms_w_text = prepare_for_cohort(
            cohort_df=task2df[task], out_dir=args.OUT_DIR,
            cohort_name=task2p[task], input_window=task2w[task], MIMIC_DIR=args.MIMIC_DIR
        )
        old_df = task2df[task]
        new_df = old_df[old_df.HADM_ID.isin(hadms_w_text)]
        print(f'Num of stays reduced from {len(old_df)} to {len(new_df)} for {task}, updated in {args.COHORT_DIR}\n')
        col = new_df.columns.tolist()[:2] + new_df.columns.tolist()[-1:]
        new_df[col].to_csv(os.path.join(args.COHORT_DIR, f'new-{task}.csv'), index=False)

    for task in ['los']:
        hadms_w_text = prepare_for_cohort(
            cohort_df=task2df[task], out_dir=args.OUT_DIR,
            cohort_name=task2p[task], input_window=task2w[task], MIMIC_DIR=args.MIMIC_DIR
        )
        for task in ['los', 'drg']:
            old_df = task2df[task]
            new_df = old_df[old_df.HADM_ID.isin(hadms_w_text)]
            print(f'Num of stays reduced from {len(old_df)} to {len(new_df)} for {task}, updated in {args.COHORT_DIR}\n')

            if task == 'drg':
                # encode drg labels
                drgs = np.sort(new_df.DRG_CODE.unique())
                assert len(drgs) == 570, 'modify task2space accordingly'
                d2l = {}
                for d in drgs:
                    d2l[d] = len(d2l)
                new_df = new_df.copy()
                new_df['DRG'] = new_df['DRG_CODE'].apply(lambda d: d2l[d])

            col = new_df.columns.tolist()[:2] + new_df.columns.tolist()[-1:]
            new_df[col].to_csv(os.path.join(args.COHORT_DIR, f'new-{task}.csv'), index=False)
     

    # print('clean four cohorts by removing hadms with no text found')
    # for task in  ['mort', 'readm', 'drg', 'los']:
    #     old_df = pd.read_csv(os.path.join(args.COHORT_DIR, f'cohort-{task}.csv'))
    #     new_df = old_df[old_df.HADM_ID.isin(cleaned_hadms)]
    #     print(f'Num of stays reduced from {len(old_df)} to {len(new_df)} for {task}')
    #     new_df.to_csv(os.path.join(args.COHORT_DIR, f'new-{task}.csv'), index=False)

