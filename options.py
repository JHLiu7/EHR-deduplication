import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--COHORT_DIR", default='data/cohorts', type=str)
parser.add_argument("--TEXT_RAW_DIR", default='data/text_cohorts_raw', type=str)
parser.add_argument("--TEXT_DEDUP_DIR", default='data/text_cohorts_dedup', type=str, help='Pickled data kept text in sentences')
parser.add_argument('--OUTPUT_DIR', type=str, default='', help='optional')

#### task options
parser.add_argument("--TASK", type=str, choices=['mort', 'drg', 'los', 'readm'])
parser.add_argument('--drop_type', type=str, default='', help='ie drop radiology or radiology+other')
parser.add_argument('--max_length', type=int, default=2500)
parser.add_argument('--max_note_length', type=int, default=450, help='based on 80 quantile')
parser.add_argument('--max_sent_num', type=int, default=40, help='sent per doc') 
parser.add_argument('--max_doc_num', type=int, default=40, help='based on 80 quantile')


#### dedup options
parser.add_argument('--skip_dedup', action="store_const", const=True, default=False, help='only preprocess text')
parser.add_argument("--skip_style", type=str, choices=['head', 'tail', 'headtail'])

parser.add_argument('--dedup_method', type=str, default='line set', choices=['line set', 'jnote'], help='jaccard, seq match archived')
parser.add_argument('--dedup_by_type', action="store_const", const=True, default=False, help='dedup within each of four note categories')
parser.add_argument('--jthresh', type=float, default=0.55)
parser.add_argument('--jthresh_glob', type=float, default=0.2)
parser.add_argument('--jngram', type=int, default=1)

parser.add_argument('--INPUT_TYPE', default='', type=str, choices=['original', 'dedupCont', 'dedupNote'], help='quick flag; do not use it when note hp change')

#### model options
parser.add_argument("--MODEL_TYPE", default='w2v', type=str, choices=['bert', 'w2v', 'hier'])
parser.add_argument('--encoder_type', type=str, default='', choices=['cnn', 'rnn'], help='encoder for hier')

parser.add_argument('--from_json_args', type=str, default='', help='json path to load train args')
parser.add_argument("--save_ckpt", action="store_const", const=True, default=False)
parser.add_argument('--from_ckpt', type=str, default='')
parser.add_argument('--from_ckpt_select', type=str, default='', help='optional')

parser.add_argument('--filter_num', type=int, default=256)
parser.add_argument('--filter_sizes', type=str, default='3,5,7')
parser.add_argument('--hidden_state', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=.2)
parser.add_argument('--freeze_emb', action="store_const", const=True, default=False)

parser.add_argument('--bert_model_name', type=str, default='bert', choices=['bert', 'clinical', 'roberta', 'longformer'])
parser.add_argument('--bert_max_length', type=int, default=512)


#### train options
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup', action="store_const", const=True, default=False)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('--fp16', action="store_const", const=True, default=False)
parser.add_argument('--grad_ckpt', action="store_const", const=True, default=False)

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--silent', action="store_const", const=True, default=False)
parser.add_argument('--checkpoint_path', type=str, default='result')

parser.add_argument('--seeds', type=str, default='')
parser.add_argument('--balance_df', default=0, type=int)
parser.add_argument("--load_no_config", action="store_const", const=True, default=False)
parser.add_argument("--debug", "-d", action="store_const", const=True, default=False)

parser.add_argument("--do_train", action="store_const", const=True, default=False)
parser.add_argument("--do_eval", action="store_const", const=True, default=False)


#### hp tune options
parser.add_argument('--num_samples', default=2, type=int)
parser.add_argument('--gpus_per_trial', default=0.5, type=float)
parser.add_argument("--on_slurm", action="store_const", const=True, default=False)

args = parser.parse_args()


if args.INPUT_TYPE == 'original':
    args.skip_dedup = True
elif args.INPUT_TYPE == 'dedupCont':
    args.dedup_method = 'line set'
elif args.INPUT_TYPE == 'dedupNote':
    args.dedup_method = 'jnote'
