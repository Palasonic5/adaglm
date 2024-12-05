import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import numpy as np

def text_process(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(p_text.split('\n\r'))
    p_text = ' '.join(p_text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text

data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
sub_domain = 'Mathematics'

jsonl_file = f'{data_dir}/{sub_domain}/papers.json'
# data = load_data(jsonl_file)
with open(jsonl_file, 'r') as file:
    data = [json.loads(line) for line in file]
i = 0
c = 2

with open(f'{data_dir}/{sub_domain}/node_text_abs_full.txt','w') as file:
    for line in tqdm(data):
        paperid = line['paper']
        # text_title = text_process(line['title'])
        # print('text title:', text_title)
        if 'abstract' in line:
            tmp_text = text_process(line['title'] + ' ' + line['abstract'])
            # abst = text_process(line['abstract'])
        else:
            tmp_text = text_process(line['title'])
        
        # if i > c: 
        #     break
        # i += 1

        file.write(f"{paperid} {tmp_text}\n")

    
