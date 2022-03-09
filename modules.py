import math, time, os, glob, random, sys
import pandas as pd 
import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, BertModel, RobertaModel, LongformerModel
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, LongformerForSequenceClassification

from utils import _task2space, _task2metric
from layers import FlatModel, HANModelRNN, HANModelCNN


def _init_trainer(args):
    ### ckpt early stopped model wts 
    log_name = args.TASK + args.MODEL_TYPE
    version = random.randint(0,5000)

    log_dir = os.path.join(args.checkpoint_path, 'log', log_name)
    model_dir= os.path.join(args.checkpoint_path, 'model', log_name, f'version_{version}')

    multigpu, auto_select_gpus = False, False
    if type(args.device) == str:
        if 'auto' in args.device:
            auto_select_gpus=True
            device = int(args.device[-1])
            multigpu = True if device >1 else False
        elif ',' in args.device:
            device = [int(d) for d in args.device.split(',')]
            if len(device) > 1:
                multigpu = True
        else:
            device = [int(args.device)]
    elif type(args.device) == int:
        device = None if args.device==-2 else [args.device]
    precision = 16 if args.fp16 else 32
    metric = 'valid_score'
    crit_mode = 'max' if args.TASK != 'los' else 'min'

    logger = TensorBoardLogger(save_dir=log_dir, version=version, name=str(log_name))

    if args.warmup:
        callbacks, epochs = [], args.epochs
    else:
        callbacks, epochs = [EarlyStopping(
            monitor=metric, min_delta=0., patience=args.patience,
            verbose=False, mode=crit_mode)], 200

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, filename='{epoch}-{%s:.3f}' % metric, 
        monitor=metric, mode=crit_mode, save_weights_only=True, 
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger, callbacks=callbacks,
        gpus=device, num_sanity_val_steps=0, max_epochs=epochs if not args.debug else 2, 
        progress_bar_refresh_rate=1-int(args.silent), accelerator='ddp' if multigpu else None,
        accumulate_grad_batches=args.accumulate_grad_batches, precision=precision, auto_select_gpus=auto_select_gpus
    )
    return trainer


##############################################################################################################
## pl modules 


class ModuleTemplate(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        if self.Y == 1:
            y = y.float()
        y_hat, y_pred = self.forward(x)
        loss = self.loss_module(y_hat,y)

        # score = self.train_metric(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        # self.log('train_score', score)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        if self.Y == 1:
            y = y.float()
        y_hat, y_pred = self.forward(x)
        loss = self.loss_module(y_hat,y)

        self.log('valid_loss', loss)
        return {'loss': loss, 'y_hat':y_hat, 'y':y}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, y_pred = self.forward(x)
        loss = self.loss_module(y_hat,y)

        self.log('test_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_epoch_end(self, list_of_dict):
        y_hat, y = [], []
        for d in list_of_dict:
            y_hat.append(d['y_hat'])
            y.append(d['y'])
        y_hat, y = torch.cat(y_hat),  torch.cat(y)
        y_pred = self._get_y_pred(y_hat)

        score= self.valid_metric(y_pred, y)
        self.log('valid_score', score, sync_dist=True)

    def _get_y_pred(self, y_hat):
        if self.Y == 1:
            # regression
            y_pred = y_hat.squeeze()
        elif self.Y == 2:
            # auroc
            y_pred = torch.softmax(y_hat, -1)[:, -1]
        else:
            # drg
            y_pred = torch.argmax(y_hat, -1)
        return y_pred

    def _setup_metric_loss(self, task):
        metric = _task2metric(task)
        self.train_metric = metric.clone()
        self.valid_metric = metric.clone()
        self.test_metric = metric.clone()

        if self.Y == 1:
            self.loss_module = nn.MSELoss()
        else:
            self.loss_module = nn.CrossEntropyLoss()


class FlatClassifier(ModuleTemplate):
    def __init__(self, args) -> None:
        super().__init__()

        # set up task
        task = args.TASK
        self.Y = _task2space(task)
        self._setup_metric_loss(task)

        # load pretrained embed
        embed_path = os.path.join(args.TEXT_RAW_DIR, 'embedding.npy')
        embeddings = np.load(embed_path)
        args.vocab_size, args.embed_dim = embeddings.shape

        args.learning_rate = args.lr

        self.save_hyperparameters(args)

        # init model
        self.model = FlatModel(Y=self.Y, **dict(self.hparams))
        
        self.model.encoder.init_embeddings(embeddings=embeddings)
        if args.freeze_emb:
            self.model.encoder.freeze_embeddings(True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, x):
        y_hat = self.model(x)
        if self.Y == 1:
            y_hat = torch.relu(y_hat)
        y_pred = self._get_y_pred(y_hat)
        return y_hat, y_pred


class HANClassifier(ModuleTemplate):
    def __init__(self, args) -> None:
        super().__init__()

        # set up task
        task = args.TASK
        self.Y = _task2space(task)
        self._setup_metric_loss(task)

        # load pretrained embed
        embed_path = os.path.join(args.TEXT_RAW_DIR, 'embedding.npy')
        embeddings = np.load(embed_path)
        args.vocab_size, args.embed_dim = embeddings.shape

        args.learning_rate = args.lr

        self.save_hyperparameters(args)

        # init model
        if args.MODEL_TYPE == 'hier':
            if args.encoder_type == 'rnn':
                self.model = HANModelRNN(Y=self.Y, **dict(self.hparams))
            else:
                self.model = HANModelCNN(Y=self.Y, **dict(self.hparams))

        elif args.MODEL_TYPE == 'hier3':
            # self.model = HANModelThree(Y=self.Y, **dict(self.hparams))
            pass
        else:
            raise NotImplementedError

        self.model.encoder.init_embeddings(embeddings=embeddings)
        if args.freeze_emb:
            self.model.encoder.freeze_embeddings(True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.wd)
        return optimizer
    
    def forward(self, x):
        y_hat = self.model(*x)
        if self.Y == 1:
            y_hat = torch.relu(y_hat)
        y_pred = self._get_y_pred(y_hat)
        return y_hat, y_pred




def _get_model_cls(model_name, grad_ckpt, Y):
    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=f'PTM/{model_name}', gradient_checkpointing=grad_ckpt, num_labels=Y)
    elif model_name == 'clinical':
        model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=f'PTM/{model_name}', gradient_checkpointing=grad_ckpt, num_labels=Y)
    elif model_name == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", cache_dir=f'PTM/{model_name}', gradient_checkpointing=grad_ckpt, num_labels=Y)
    elif model_name == 'longformer':
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096", cache_dir=f'PTM/{model_name}', gradient_checkpointing=grad_ckpt, num_labels=Y)
    else:
        raise NotImplementedError('lack of bert clf implementation')
    return model


class BERTClassifier(ModuleTemplate):
    def __init__(self, args) -> None:
        super().__init__()

        args.learning_rate = args.lr
        self.save_hyperparameters(args)

        task = args.TASK
        self.Y = _task2space(task)

        ## set up model
        self.model = _get_model_cls(model_name=args.bert_model_name, grad_ckpt=args.grad_ckpt, Y=self.Y)

        ## to use warmup or not
        if self.hparams.warmup is True:
            n_samples = args.n_samples # need to be specified
            epochs = self.hparams.epochs if not args.debug else 2
            step_total = ((n_samples // self.hparams.batch_size) // self.hparams.accumulate_grad_batches) * epochs
            self.step_total = int(step_total)
            self.step_warmup= math.ceil(step_total*0.1)

        ## set up metric and loss
        self._setup_metric_loss(task)
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.wd,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        if self.hparams.warmup:
            scheduler = get_linear_schedule_with_warmup(optimizer, self.step_warmup, self.step_total)
            print(f'*****Setting up warmup scheduler, with {self.step_total} total steps and {self.step_warmup} warmup steps*****')
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        y_hat = self.model(**x).logits
        if self.Y == 1:
            y_hat = torch.relu(y_hat)
        y_pred = self._get_y_pred(y_hat)
        return y_hat, y_pred

