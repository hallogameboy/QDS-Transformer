README for QDS-Transformer Implementation
EMNLP 2020 Submission ID: 3388
Title: Long Document Ranking with Query-Directed Sparse Transformer

---

Required Packages and Dependencies

* Python 3.6.9 (or a compatible version)
* PyTorch 1.2.0 (or a compatible version)
* Abseil 1.13.0 (or a compatible version)
* NumPy 1.17.4 (or a compatible version)
* scikit-learn 0.22 (or a compatible version)
* Yaml 5.2 (or a compatible version)
* ujson (or be subsequently replaced with geuine json)
* tqdm 4.40.1 (or a compatible version)
* NLTK 3.3 (or a compatible version)
* pytrec_eval


---

Data Format


* The data format of qrels, queries, and top100 files follow the official TREC
 formats. Take the TREC-19 DL track as an example, most of the files can be
 directly utilized.

* For document collection the corpus, to indicate the sentence structure, we
 employ NLTK to segment each document into sentences and store it in a JSON
 format as:

...
{
    'docid': DOCID,
    'ss':[SENT1, SENT2, ...]
}
...

Specifically, each line represents a document.

* For the official tsv-based data format, we also prepare a script to convert
 the data. The document collection `/PATH/TO/msmarco-docs.tsv` would be
 converted to `/PATH/TO/msmarco.docs.tsv.sentences` with the following script:

./split_collection_into_sentences.py /PATH/TO/msmarco-docs.tsv


---

Experimental Configuration

* The file `config.yml` declares the required paths and naming for
  experiments.

** path_data, path_model, and path_tb denote the directories for datasets,
   trained model storage, and tensorboard results.

** corpus represents the file name of processed document collection.

** data_prefix indicates the prefix string of the dataset for different tasks.


---

Dataset Preparation

* Experiments require three sets for train, dev, and test datasets in the data
  folder following the corresponding set names and the prefix constraint.

* More precisely, given the data_prefix as "msmarco-doc", we need to have the
  following 9 files for experiments following the TREC formats.

msmarco-doctrain-queries.tsv
msmarco-doctrain-qrels.tsv
msmarco-doctrain-top100

msmarco-docdev-queries.tsv
msmarco-docdev-qrels.tsv
msmarco-docdev-top100

msmarco-doctest-queries.tsv
msmarco-doctest-qrels.tsv
msmarco-doctest-top100


---

Pretrained Model

* We train QDS-Transformer from a Longformer pretrained model.

* The pretrained model can be obtained from
https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz

* The pretrained model should be placed at the same folder to the training
  script.


---

Training Process

* While the dataset is ready, the following script can initiate the training
  process:

CUDA_VISIBLE_DEVICES=0 python3 train.py \
--config=config.yml \
--model_prefix=<model_prefix> \
--tb \
--tb_per_iter=<ITER>

* For every <ITER> iterations, the script will evaluate the model at the
  moment on the dev set, if its dev MRR@10 is better than previously best
  model, we stores the model at path_model declared in config.yml. Also, we
  will evaluate the test performance and generate a TREC format prediction run
  for further formal evaluation with trec_eval. Note that the prediction files
  can be found in the path_model.

* Besides, all of the computations will be recorded in path_tb declared in
  config.yml for the Tensorboard usage.

* More feasible arguments can be referred to the utils.py file.


---

Inference Process

* To evalute the test dataset with an arbitrary model, the following script
  can execute the inference process:


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
--config=config.yml \
--load_model=/PATH/TO/MODEL \
--pred_file=/PATH/TO/PREDICTION

* /PATH/TO/MODEL is directed to the path of trained model from the training
  process.

* /PATH/TO/PREDICTION assigns the path for dumping predictions of data in the
  test set.
