import os, sys, warnings, logging, json


import torch
import pytorch_lightning as pl


from modules import BERTClassifier, FlatClassifier, HANClassifier, _init_trainer
from modules_data import TextDedupDataModule 

from utils import inference, score_inf
from options import args

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

if __name__ == '__main__':

    if os.path.isfile(args.from_json_args):
        json_args = json.load(open(args.from_json_args, 'r'))
        vars(args).update(json_args)
        logging.info(f'Loaded hp: {str(json_args)}')

    ## balance df for readm
    if args.TASK == 'readm':
        args.balance_df = 1

    logging.info('Loading data..')
    dm = TextDedupDataModule(args)
    dm.setup()
    logging.info('Data loaded')

    if args.warmup:
        args.n_samples = len(dm.train_dataset)

    logging.info('Loading model..')
    findModel = {
        'w2v': FlatClassifier,
        'bert': BERTClassifier,
        'hier': HANClassifier,
        'hier3': HANClassifier
    }
    

    if args.from_ckpt != '':
        model_dir = args.from_ckpt
        json_args = json.load(open(os.path.join(model_dir, 'hp.json'), 'r'))

        if 'w2v' in model_dir:
            args.MODEL_TYPE = 'w2v'
        elif 'hier' in model_dir:
            args.MODEL_TYPE = 'hier'
        
        vars(args).update(json_args)
        model = findModel[args.MODEL_TYPE](args)

        wts = torch.load(os.path.join(model_dir, 'model.bin'))
        model.model.load_state_dict(wts)
        logging.info(f'Model wts loaded from {model_dir}')
    else:
        model = findModel[args.MODEL_TYPE](args)
        logging.info('Model loaded')


    trainer = _init_trainer(args)

    if args.do_train:
        trainer.fit(model, dm)
        trainer.test(ckpt_path='best')

    if args.do_eval:
        logging.info('Evaluate')

        dataloaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
        model = model.cuda()

        infs, splits = {}, ['train', 'val', 'test']
        for split, dataloader in zip(splits, dataloaders):
            output = inference(model, dataloader)
            infs[split] = output

            score_dict, line = score_inf(args.TASK, output)

            print(split, 'cases:', len(output['stay']))
            print(line, '\n') 
            if split == 'test':
                print('all scores:', score_dict)



    logging.info('Finished')