#!/usr/bin/env python3
import subprocess
import sys
import multiprocessing as mp
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
try:
    import ujson as json
except:
    import json


'''
time ./split_collection_into_sentences.py ../../data/trec19/msmarco-docs.tsv

real    21m8.941s
user    254m3.632s
sys     18m55.108s
'''

NUM_THREADS = 32
if __name__ == '__main__':
    docset = set()
    if len(sys.argv) < 1 + 1:
        print('--usage {} corpus_file'.format(sys.argv[0]),
                file=sys.stderr)
        sys.exit(0)

    file_name = sys.argv[1]
    
    def preprocess(line):
        docid, url, title, body = line.split('\t')
        sents = [title] + sent_tokenize(body)
        return [docid, sents]

    lc = int(subprocess.check_output(
        'wc -l {}'.format(file_name),
        shell=True).split()[0])
    with open(file_name, 'r') as fp:
        lines = [x for x in tqdm(fp, desc='Loading', total=lc)]

    with mp.Pool(NUM_THREADS) as p:
        results = list(tqdm(p.imap(preprocess, lines),
            desc='Processing', total=lc))

    with open(file_name + '.sentences', 'w') as wp:
        for docid, ss in tqdm(results, 'Dumping', total=lc):
            print(json.dumps({'docid': docid, 'ss': ss}), file=wp)

