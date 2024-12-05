import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import numpy as np
# from transformers import BertTokenizerFast

random.seed(0)
data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
sub_dataset = 'cloth'

# read raw data
with open(f'{data_dir}/{sub_dataset}/product.json') as f:
    raw_data = {}
    readin = f.readlines()
    for line in tqdm(readin):
        #data.append(json.loads(line))
        #data.append(eval(line.strip()))
        tmp = eval(line.strip())
        raw_data[tmp['asin']] = tmp
#random.shuffle(data)

# statistics on label names
label_name_stat = defaultdict(int)

for did in tqdm(raw_data):
    sample = raw_data[did]
    c_list = list(set(sum(sample['categories'], [])))
    for c in c_list:
        label_name_stat[c] += 1

# read label name dict

label_name_dict = {}
label_name_set = set()
label_name2id_dict = {}

for n in label_name_stat:
    if label_name_stat[n] > int(0.5 * len(raw_data)):
        continue

    label_name_dict[len(label_name_dict)] = n
    label_name_set.add(n)
    label_name2id_dict[n] = len(label_name_dict) - 1

print(f'Num of unique labels:{len(label_name_set)}')

# filter item with no text

data = {}
for idd in tqdm(raw_data):
    if 'title' in raw_data[idd]:
        data[idd] = raw_data[idd]
print(len(data))

# text processing function
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

# save node texts
with open(f'{data_dir}/{sub_dataset}/node_text.txt','w') as file:
    for idd in tqdm(data, desc="Processing Data"):
        # print(idd)
        if 'description' in data[idd]:
            tmp_text = text_process(data[idd]['title'] + ' ' + data[idd]['description'])
        else:
            tmp_text = text_process(data[idd]['title'])
            
        file.write(f"{idd} {tmp_text}\n")

with open(f'{data_dir}/{sub_dataset}/processed_data.json', 'w') as file:
    for idd in tqdm(data):
        json_line = json.dumps(data[idd])
        file.write(json_line + '\n')

