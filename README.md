# Deduplicate EHR notes for downstream tasks

This repository contains research code to analyze the impact of text redundancy on four common early prediction tasks in the clinical scenario, including in-hospital mortality, remaining hospital length-of-stay (LOS), discharged diagnostic-related group (DRG), and 30-day hospital readmission predictions. 

## 0. Setup

The code is based on python 3.6.9, though higher versions should be compatible as well. Main packages required to reproduce the results include:

- pandas==1.1.4
- torch==1.4.0
- pytorch-lightning==1.3.8

A complete list of packages can be found in `requirements.txt` to create a virtual environment. 

## 1. Prepare data

### 1.1 Cohort extraction

Run the `sql` files on BigQuery to obtain the raw cohorts for the four examined tasks. This requires credential approved for accessing the MIMIC-III database. Then one can query the database following steps described [here](https://mimic.mit.edu/docs/gettingstarted/cloud/) and [here](https://mimic.mit.edu/docs/iii/tutorials/intro-to-mimic-iii-bq/). Then save the queried results in `csv` format and put under the same folder. For example, run `cohort-mort.sql` on BigQuery and then save the output as `cohort-mort.csv` in `data/cohorts`. 

### 1.2 Note preparation

Run `prepare_raw.py` as below to extract notes for the cohorts, which will also perform extra filtering of cohorts (i.e., removing stays w/o any written notes). This requires `NOTEEVENTS.csv.gz` be downloaded.

```sh
cd data/

nice python prepare_raw.py --MIMIC_DIR $PATH_TO_MIMIC_DATA_FOR_NOTEEVENTS
```

### 1.3 Word embedding

In this work we leveraged the BioWordVec word embeddings. If you'd like to do the same, download the embedding from their [repo](https://github.com/ncbi-nlp/BioWordVec) and run below

```
nice python prepare_word_emb.py --pretrained_embed_dir $PATH_TO_BIOWORD_VEC 
```

## 2. Text deduplication

Here we prepared deduplicated versions of input as a separate step to enable faster modeling iterations, but it is also possible to skip this step and perform deduplication on the fly. 

```sh
nice python deduper.py --n_jobs $NUM_OF_PROCESS --TEXT_DEDUP_DIR data/text_cohorts_dedup
```

## 3. Hyperparameter tuning

To compare the impact of different input settings as fairly as possible, we tuned the hyper parameters for the CNN-based medium and full-context models, as described in the paper. Here is a sample code to run our tuning experiments, which relies on `ray[tune]`:

```sh
python tune.py \
    --device '4' \
    --max_length 2500 \
    --max_note_length 1000 \
    --MODEL_TYPE w2v \
    --batch_size 16 \
    --gpus_per_trial 0.25 \
    --num_samples 50 \
    --TASK mort \
    --INPUT_TYPE original \
    --OUTPUT_DIR runs/best_configs/flat
```

Tuning for all task-input-model trios is a time-consuming process. Here we release the best hyperparameter configuration for each setting in the folder `best_configs` hosted on [gdrive](https://drive.google.com/drive/folders/1zth1kWeWz4FURz6r5ClP594o8uXSCBdF?usp=sharing). 

## 4. Training and evaluation

New models can be trained and evaluated using command like: 

```sh
python run.py --do_train --do_eval --TASK los --INPUT_TYPE dedupCont --MODEL_TYPE hier
```

We also release the best checkpoints we used to report results for each experiment in our paper, which can be found in `ckpt` on [gdrive](https://drive.google.com/drive/folders/1zth1kWeWz4FURz6r5ClP594o8uXSCBdF?usp=sharing). Then can be used for evaluation by running command like:

```sh
python run.py --do_eval --TASK mort --INPUT_TYPE original --MODEL_TYPE hier --from_ckpt ckpt/full-hier-cnn/mort_nodedup_hier_1000doc40_0.852/ 
```



### Acknowledgement

We would like to thank the people and communities that released the code/packages or created the resources we used in this repository. 
