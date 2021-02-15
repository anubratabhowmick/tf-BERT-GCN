## Introduction

We propose the extension of the model proposed by Jeong, Jang & Park (2020) by combining the context, citation history with co-authorship information into the recommendation system. We also propose to use more domain-specific embeddings to better capture the semantics in the context. Our experiments show the positive effect of co- authorship information on citation recommendations, and that our model based on the combination of domain- specifically embedded context, the citation and the co-authorship history significantly outperforms the basic context-based recommendation model.

Our code is based on that([BERT](https://github.com/google-research/bert), [GCN](https://github.com/tkipf/gae/)).

## Data
- [Full Context PeerRead](https://bert-gcn-for-paper-citation.s3.ap-northeast-2.amazonaws.com/PeerRead/full_context_PeerRead.csv) : Created by processing [allenai-PeerRead](https://github.com/allenai/PeerRead)

In order o run the tensorflow version of the BERT-GCN Square mode:

```
1. Download the repository to your local machine.
2. Download [Full Context PeerRead](https://bert-gcn-for-paper-citation.s3.ap-northeast-2.amazonaws.com/PeerRead/full_context_PeerRead.csv) dataset
3. Save the dataset in /glue/ACRS folder.
4. Create a folder pre_train with subfolders gcn, BERT-base, and BERT_Sci_Base.
5. Download the pretrained [BERT-base](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) and save in BERT-base folder, and the [Sci-BERT base](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz) and save in BERT_Sci_Base.
6. First run the gcn_pretrain.py using the command python gcn_pretrain.py  --dataset PeerRead --gcn_hidden1 9529 --paper_author author
  ```
    gcn_hidden1 = 9529 and paper_author = author when you want to train the co-authorship network.
    gcn_hidden1 = 4837 and paper_author = paper when you want to train the citation network.
    The parameter descriptions can be found [here](https://github.com/tkipf/gae).
      * `--gcn_model`, `--gcn_lr`, `--gcn_epochs`, `--gcn_hidden1`, `--gcn_hidden2`, ... 
  ```
7. After the citation and co-authorship networks are generated, you need to run the run_classifier.py with the following commands:
  ```
    PeerRead (with Sci-BERT): python3 run_classifier.py --model=bert_gcn --dataset=PeerRead --do_train=true --do_predict=true \
    --data_dir=./glue/ACRS --vocab_file=./pre_train/BERT_Sci_Base/vocab.txt \
    --bert_config_file=./pre_train/BERT_Sci_Base/bert_config.json \
    --init_checkpoint=./pre_train/BBERT_Sci_Base/bert_model.ckpt \
    --max_seq_length=50 --train_batch_size=16 --learning_rate=2e-5 \
    --num_train_epochs=30.0 --output_dir=./output --frequency=5 --year 2017
    
    PeerRead (with BERT base): ython3 run_classifier.py --model=bert_gcn --dataset=PeerRead --do_train=true --do_predict=true \
    --data_dir=./glue/ACRS --vocab_file=./pre_train/BERT-base/vocab.txt \
    --bert_config_file=./pre_train/BERT-base/bert_config.json \
    --init_checkpoint=./pre_train/BERT-base/bert_model.ckpt \
    --max_seq_length=50 --train_batch_size=16 --learning_rate=2e-5 \
    --num_train_epochs=30.0 --output_dir=./output --frequency=5 --year 2017
    
    * General Parameters:
    * `--model` (Required): The mode to run the `run_classifier.py` script in. Possible values: `bert` or `bert_gcn`
    * `--dataset` (Required): The dataset to run the `run_classifier.py` script in. Possible values: `AAN` or `PeerRead`
    * `--frequency` (Required): Parse datasets more frequently
    * `--max_seq_length` : Length of cited text to use 
    * `--gpu` : The gpu to run code

    * BERT Parameters: You can refer to it [here](https://github.com/google-research/bert).
    * `--do_train`, `--do_predict`, `--data_dir`, `--vocab_file`, `--bert_config_file`, `--init_checkpoint`, ...
  ```
```
## Result
Our result from the GCN Square are shown below:
![Alt text](./images/result.png?raw=true "Result")
