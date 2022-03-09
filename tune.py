import os, warnings, logging, math, json
from datetime import datetime


import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from modules import BERTClassifier, FlatClassifier, HANClassifier
from modules_data import TextDedupDataModule
from utils import _name4tunejob
from options import args

warnings.filterwarnings("ignore")
os.environ["SLURM_JOB_NAME"] = "bash"

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

def train_with_tune(config, args, dm, num_gpus):
    ## fix dropout name
    if 'zdropout' in config:
        config['dropout'] = config.pop('zdropout')

    ## optinally reduce bsize given run config (effect bsize kept same by accumulating grad)
    bsize = args.batch_size
    accumulate_grad = 1

    if args.MODEL_TYPE == 'w2v':
        cond1 = len(config['filter_sizes']) > 5
        cond2 = config['filter_num'] >= 384
        cond3 = config['num_layers'] > 1
        if (cond1 and cond2) or (cond2 and cond3):
            bsize = int(args.batch_size / 2)
            accumulate_grad = 2

    if args.MODEL_TYPE == 'hier':
        cond1 = config['filter_num'] > 384
        if args.TASK == 'readm':
            cond1 = config['filter_num'] >= 384
            
        if cond1:
            bsize = int(args.batch_size / 2)
            accumulate_grad = 2
        
    ## load model 
    vars(args).update(config)

    logging.info('Loading model..')
    findModel = {
        'w2v': FlatClassifier,
        'hier': HANClassifier
    }
    model = findModel[args.MODEL_TYPE](args)
    logging.info('Model loaded')

    metric = 'valid_score'
    stop_mode = 'max' if args.TASK != 'los' else 'min'

    ## get training going
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            EarlyStopping(monitor=metric, min_delta=0.00, patience=args.patience, verbose=False, mode=stop_mode),
            TuneReportCallback({"mean_score": metric}, on='validation_end')
        ],
        max_epochs=200 if not args.debug else 2, gpus=math.ceil(num_gpus), 
        progress_bar_refresh_rate=0, accumulate_grad_batches=accumulate_grad
    )

    train_dataloader = dm.train_dataloader(bsize)
    val_dataloader = dm.val_dataloader(bsize)

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':

    # --------
    # prepare
    # --------

    ## start 
    start = datetime.now()
    
    ## balance df for readm
    if args.TASK == 'readm':
        args.balance_df = 1


    ## fix arg flags; change this accordingly
    local_dir = '/home/jinghuil1/TmpDedupEHR/'
    if args.on_slurm:
        local_dir = '/data/gpfs/projects/punim0478/jinghuil1/TmpDedupEHR/'

    args.COHORT_DIR, args.TEXT_RAW_DIR, args.TEXT_DEDUP_DIR = [os.path.join(local_dir, path) for path in (args.COHORT_DIR, args.TEXT_RAW_DIR, args.TEXT_DEDUP_DIR)]

    ## load data
    logging.info('Loading data..')
    dm = TextDedupDataModule(args)
    dm.setup()
    logging.info('Data loaded')

    root_outpath = 'runs/tune/hier/best_configs/' if args.MODEL_TYPE == 'hier' else 'runs/tune/flat/best_configs/'
    if args.OUTPUT_DIR != '':
        root_outpath = args.OUTPUT_DIR

    ## load/specify configs
    if os.path.isfile(args.from_json_args):
        # grid search
        json_args = json.load(open(args.from_json_args, 'r'))

        ## put dropout last
        json_args['zdropout'] = json_args.pop('dropout')

        configs = {} 
        for k, v in json_args.items():
            assert type(v) == list
            configs[k] = tune.grid_search(v)
        num_samples = 1

        hp_path = os.path.join(root_outpath, 'grid')
    else:
        # random search
        configs = {
            "zdropout": tune.uniform(0.2, 0.5),
            "lr": tune.loguniform(1e-4, 5e-3),
        }
        if args.MODEL_TYPE == 'w2v':
            configs.update({
                "filter_num": tune.qrandint(128, 512, 128),
                "filter_sizes": tune.choice(['3,5', '1,3,5', '3,5,7', '3,5,7,9']),
                "num_layers": tune.choice([1, 2])
            })
        elif args.MODEL_TYPE == 'hier':
            configs.update({
                "filter_num": tune.qrandint(128, 512, 128),
                "filter_sizes": tune.choice(['5-1', '7-1', '3,5-1', '3,5-3', 
                                '3,5,7-1', '3,5,7-3', '3,5-1,3', '3,5,7-1,3']),
            })
        num_samples = args.num_samples

        hp_path = os.path.join(root_outpath, f'random_{num_samples}')

    os.makedirs(hp_path, exist_ok=True)



    # --------
    # tune
    # --------
    select_mode = 'max' if args.TASK != 'los' else 'min'
    asha_scheduler = ASHAScheduler(max_t=50, grace_period=5, reduction_factor=2)

    trial_name = f'tune_with_asha_{args.MODEL_TYPE}'
    analysis = tune.run(
        tune.with_parameters(train_with_tune, args=args, dm=dm, num_gpus=args.gpus_per_trial),
        resources_per_trial={'cpu':1, 'gpu':args.gpus_per_trial},
        metric='mean_score', mode=select_mode, config=configs,
        num_samples=num_samples, name=trial_name, scheduler=asha_scheduler, local_dir=local_dir
    )

    best_config = analysis.best_config
    if 'zdropout' in best_config:
        best_config['dropout'] = best_config.pop('zdropout')


    best_score = analysis.best_result['mean_score']
    line1 = "Best hyperparameters found were: " + str(best_config)
    line2 = "Best val score: {:.4f}".format(best_score)
    logging.info(line1)
    logging.info(line2)


    # --------
    # save hp
    # --------
    hp_name = _name4tunejob(args)
    hp_name = hp_name.replace('.json', f'_val{best_score:.2f}.json')

    with open(os.path.join(hp_path, hp_name), 'w') as f:
        json.dump(best_config, f)


    print('\n', args.TASK, args.MODEL_TYPE, hp_name)
    print(line1)
    print(line2)

    time = (datetime.now() - start).total_seconds()
    print(f"Finished in {int(time//3600)} hrs {(time%3600)/60:.2f} min")

