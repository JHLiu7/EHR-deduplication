import pandas as pd 
import numpy as np 
import pickle as pk
import string, re, torch, os
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import torchmetrics

from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve, accuracy_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def _task2target(task):
    tdict = {
        'mort': 'MORT_HOSP',
        'readm': 'READM_30',
        'drg': 'DRG',
        'los': 'LOS_DAY'
    }
    return tdict[task]

def _task2space(task):
    tdict = {
        'mort': 2,
        'readm': 2,
        'drg': 570,
        'los': 1
    }
    return tdict[task]


def _task2rawfile(task, debug=False):
    task2p = {
        'mort': 'mort', 'readm': 'readm', 'drg': 'drglos', 'los': 'drglos'
    }
    p2files = {
        'mort': 'mort_1daywindow_33515hadms_305737notes.p',
        'readm': 'readm_fullwindow_34028hadms_854570notes.p',
        'drglos': 'drglos_2daywindow_37303hadms_470727notes.p',
    }
    if debug:
        p2files = {
            'mort': 'mort_1daywindow_40hadms_300notes.p',
            'readm': 'readm_fullwindow_40hadms_816notes.p',
            'drglos': 'drglos_2daywindow_40hadms_438notes.p',
        }
    return p2files[task2p[task]]

def _cohort2sets(cohort_df, task):
    ratios = (0.1, 0.1) if task != 'drg' else (0.05, 0.1)
    train_df, val_df, test_df = split_cohort_df_by_subj(cohort_df, ratios=ratios)
    return train_df, val_df, test_df

def _name4dedup(task, dedup_args):
    # give it a name

    # no dedup
    if dedup_args['skip_dedup']:
        tp = 'raw'
        return '_'.join([task, tp]) + '.p'

    # dedup
    tp = 'notype' if dedup_args['ignore_type'] else 'wtype'
    if dedup_args['method'] not in ['jaccard', 'jnote']:
        mt = dedup_args['method'].replace(' ', '')
    else:
        if dedup_args['n_gram']>1:
            n = dedup_args['n_gram']
            n_gram = f'ngram{n}'
        else:
            n_gram = '' 

        if dedup_args['method']=='jaccard':
            mt = dedup_args['method'] + str(dedup_args['thresh']).replace('.','') + str(dedup_args['global_thresh']).replace('.','') + n_gram
        elif dedup_args['method']=='jnote':
            mt = dedup_args['method'] + str(dedup_args['thresh']).replace('.','') + n_gram

    fname = '_'.join([task, tp, mt]) + '.p'
    return fname

def _args4dedup(args):
    dedup_args = {
        'method': args.dedup_method, 'ignore_type': not args.dedup_by_type, 'skip_dedup': args.skip_dedup, 
        'thresh': args.jthresh, 'global_thresh': args.jthresh_glob, 'n_gram': args.jngram
    }
    return dedup_args

def _is_num_unit(word, limit=10):
    num = any(char.isdigit() for char in word)
    lth = len(word) < limit
    return num & lth

def _jaccard_dist(token1, token2):
    set1, set2 = [set(i) for i in (token1, token2)]
    
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return 1 - inter/union

def _strip_phi(t):
    t = re.sub(r'\[\*\*.*?\*\*\]', ' ', t)
    t = re.sub(r'_', ' ', t)
    t = re.sub(r"`", '', t)
    t = re.sub(r"''", '', t)
    t = re.sub(r'"', '', t)
    return t

NOTE_CATEGORIES = ['radiology', 'nursing', 'others', 'physician']
def _fix_category(cat):
    if cat == 'Radiology':
        ncat = 'radiology'
    elif cat == 'Nursing/other' or cat == 'Nursing':
        ncat = 'nursing'
    elif cat == 'Physician ':
        ncat = 'physician'
    else:
        ncat = 'others'
    return ncat

def split_cohort_df_by_subj(cohort_df, ratios=(0.1, 0.1), random_state_val=1234, random_state_test=1443):
    """
    split by subj
    ratios: (val, test)
    """
    subj_ct = cohort_df.SUBJECT_ID.value_counts()
    subj_sg = pd.Series(subj_ct[subj_ct==1].sort_index().index)
    subj_ml = pd.Series(subj_ct[subj_ct >1].sort_index().index)
    
    val_ratio, test_ratio = ratios
    
    # split test
    test_subj = subj_sg.sample(frac=test_ratio, random_state=random_state_test).tolist() + \
                subj_ml.sample(frac=test_ratio, random_state=random_state_test).tolist()
    
    subj_sg, subj_ml = subj_sg[~subj_sg.isin(test_subj)], subj_ml[~subj_ml.isin(test_subj)]
    
    # split val
    val_ratio = val_ratio / (1-test_ratio)
    
    val_subj  = subj_sg.sample(frac=val_ratio, random_state=random_state_val).tolist() + \
                subj_ml.sample(frac=val_ratio, random_state=random_state_val).tolist()
    
    subj_sg, subj_ml = subj_sg[~subj_sg.isin(val_subj)], subj_ml[~subj_ml.isin(val_subj)]
    
    train_subj = subj_sg.tolist() + subj_ml.tolist()
    
    # split cohort df
    train_df = cohort_df[cohort_df['SUBJECT_ID'].isin(train_subj)]
    val_df = cohort_df[cohort_df['SUBJECT_ID'].isin(val_subj)]
    test_df = cohort_df[cohort_df['SUBJECT_ID'].isin(test_subj)]
    
    assert len(train_df) + len(val_df) + len(test_df) == len(cohort_df)
    
    return train_df, val_df, test_df


##### metrics and scores

def _task2metric(task):
    tdict = {
        'mort': torchmetrics.AUROC(pos_label=1),
        'readm': torchmetrics.AUROC(pos_label=1),
        'drg': torchmetrics.Accuracy(),
        'los': torchmetrics.MeanAbsoluteError()
    }
    return tdict[task]

def _auc_scores(y_pred, y):
    auroc = roc_auc_score(y, y_pred)
    prec, rec, _ = precision_recall_curve(y, y_pred)
    aupr = auc(rec, prec)
    y_flat = np.around(y_pred, 0).astype(int)
    f1 = f1_score(y, y_flat)
    line = f'AUCROC\tAUCPR\n{auroc:.3f}\t{aupr:.3f}'
    out = {
        'auroc': auroc, 'aupr': aupr, 'f1': f1
    }
    return out, line

def _acc_f1(y_pred, y):
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro', labels=np.unique(y))
    line = f'Acc\tMacroF1\n{acc:.3f}\t{f1_macro:.3f}'
    out = {
        'acc': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro
    }
    return out, line 

def _error_corr(y_pred, y):
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    corr, p = pearsonr(y, y_pred)
    line = f'MAE\tPearsonr\n{mae:.3f}\t{corr:.3f}'
    out = {
        'mae': mae, 'corr': corr, 'pval': p, 'rmse': rmse
    }
    return out, line 

def score_inf(task, output):
    task2scores = {
        'mort': _auc_scores,
        'readm': _auc_scores,
        'drg': _acc_f1,
        'los': _error_corr
    }

    y_pred, y = output['y_pred'], output['y']
    sdict, line = task2scores[task](y_pred, y)
    
    return sdict, line

def _task2mainscore(task):
    task2main = {
        'mort': 'auroc',
        'readm': 'auroc',
        'drg': 'acc',
        'los': 'mae'
    }
    return task2main[task]

#### inf 

def inference(model, dataloader):
    use_cuda = torch.cuda.is_available()
    model.eval()

    y_hat_all, y_pred_all, y_all, stay_all = [], [], [], []
    for batch in dataloader:
        x, y, stay = batch
        if use_cuda:
            if type(x) is dict:
                x = {k:v.cuda() for k,v in x.items()}
            elif type(x) is tuple:
                x = (i.cuda() for i in x)
            else:
                x = x.cuda()

        with torch.no_grad():
            y_hat, y_pred = model(x)

        y_hat_all.append(y_hat.cpu().detach().numpy())
        y_pred_all.append(y_pred.cpu().detach().numpy())
        y_all.append(y.numpy()) 
        stay_all.append(list(stay))

    output = {
        'y_hat': np.concatenate(y_hat_all),
        'y_pred': np.concatenate(y_pred_all),
        'y': np.concatenate(y_all),
        'stay': np.concatenate(stay_all),
    }
    
    return output




def _name4tunejob(args):

    ## f'task-dedup-context-model'
    task = args.TASK
    
    # skip dedup!
    if args.skip_dedup:
        dedup = 'nodedup'
    else:
        if args.dedup_method not in ['jaccard', 'jnote']:
            dedup = args.dedup_method.replace(' ', '')
        else:
            ngram = f'ngram{args.jngram}' if args.jngram > 1 else ''
            if args.dedup_method == 'jaccard':
                dedup = args.dedup_method + str(args.jthresh).replace('.','') + str(args.jthresh_glob).replace('.','') + ngram
            elif args.dedup_method == 'jnote':
                dedup = args.dedup_method + str(args.jthresh).replace('.','') + ngram


    # context = f'maxlength{args.max_length}'
    if args.MODEL_TYPE == 'hier':
        context = f'{args.MODEL_TYPE}_{args.max_note_length}doc{args.max_doc_num}'
    elif args.MODEL_TYPE == 'hier3':
        context = f'{args.MODEL_TYPE}_{args.max_note_length}sent{args.max_sent_num}'
    else:
        context = f'{args.MODEL_TYPE}_{args.max_length}'
    
    model = args.MODEL_TYPE
    if model == 'bert':
        context = f'{args.MODEL_TYPE}_{args.bert_max_length}'

    name = '_'.join([task, dedup, context])
    if args.debug:
        name += '_sample'
    return name + '.json'

def _name4ckpt(args, score):

    ## task-inputsetting-context-model-score

    task = args.TASK

    if args.skip_dedup:
        dedup = 'nodedup'
    else:
        if args.dedup_method not in ['jaccard', 'jnote']:
            dedup = args.dedup_method.replace(' ', '')
        else:
            ngram = f'ngram{args.jngram}' if args.jngram > 1 else ''
            if args.dedup_method == 'jaccard':
                dedup = args.dedup_method + str(args.jthresh).replace('.','') + str(args.jthresh_glob).replace('.','') + ngram
            elif args.dedup_method == 'jnote':
                dedup = args.dedup_method + str(args.jthresh).replace('.','') + ngram

    if args.MODEL_TYPE == 'hier':
        context = f'{args.MODEL_TYPE}_{args.max_note_length}doc{args.max_doc_num}'
    elif args.MODEL_TYPE == 'hier3':
        context = f'{args.MODEL_TYPE}_{args.max_note_length}sent{args.max_sent_num}'
    else:
        context = f'{args.MODEL_TYPE}_{args.max_length}'
    score = f'{score:.3f}'

    name = '_'.join([task, dedup, context, score])
    if args.debug:
        name += 'sample'
    return name

