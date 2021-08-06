import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
#import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper, BartModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
import torch

import math
import random
import re
import argparse

#inc = 0

def encode_sentences(tokenizer, df, source_keys, target_key, max_length=1024, pad_to_max_length=True, return_tensors="pt"):
    
    def run_bart(snippet):
        encoded_dict = tokenizer(
          snippet,
          max_length=max_length,
          truncation=True,
          add_prefix_space = True
        )
        return encoded_dict

    def get_encoding(snippet):
        all_input_ids = []
        all_attention_masks = []
        encoded_dict = {}
        if isinstance(snippet, list):
            #snippet = [each for each in snippet if each.strip()]
            enc = run_bart(snippet)
            batch_ids = enc['input_ids']
            
            for each_list in batch_ids:
                all_input_ids += each_list

            for each_list in enc['attention_mask']:
                all_attention_masks += each_list

            if len(all_input_ids) > max_length:
                all_input_ids = all_input_ids[:1023] + [2]
                all_attention_masks = all_attention_masks[:1024]

            pad_len = max_length - len(all_input_ids)
            pad_list = [1] * pad_len
            attention_mask = [0] * pad_len
            all_input_ids += pad_list
            all_attention_masks += attention_mask
            
            input_ids = torch.as_tensor([all_input_ids])
            attention_mask = torch.as_tensor([all_attention_masks])
            encoded_dict["input_ids"] = input_ids
            encoded_dict["attention_mask"] = attention_mask
             
            bos_ids = [i for i, t in enumerate(encoded_dict['input_ids'].tolist()[0]) if t == 0]
            pad_len = max_length - len(bos_ids)
            pad_list = [-2] *  pad_len
            bos_ids = bos_ids + pad_list
            bos_ids = torch.tensor([bos_ids])
            assert(input_ids.shape == attention_mask.shape ==bos_ids.shape)
        return encoded_dict, bos_ids

    encoded_sentences = {}

    target_ids = []

    for key in source_keys:
        id_key = '%s_ids'%key
        attention_mask_key = '%s_attention_masks'%key
        bos_key = "%s_bos_ids"%key
        
        if key not in encoded_sentences:
            encoded_sentences[id_key] = []
            encoded_sentences[attention_mask_key] = []
            encoded_sentences[bos_key] = []


    for idx, row in df.iterrows():
        row_dict = {}
        all_present = True

        for key in source_keys:
            id_key = '%s_ids'%key
            attention_mask_key = '%s_attention_masks'%key
            bos_key = "%s_bos_ids"%key
            if key not in row_dict:
                row_dict[id_key] = []
                row_dict[attention_mask_key] = []
                row_dict[bos_key] = []


            sentences_key = row[key]
            sentences_key = eval(sentences_key)
            sentences_key = [each for each in sentences_key if each.strip()]
            #sentences_key = ["<%s> "%key +each + " </%s>"%key for each in sentences_key ]
            if sentences_key:
                sentence_encoding, bos_ids = get_encoding(sentences_key)

                row_dict[id_key].append(sentence_encoding['input_ids'])
                row_dict[attention_mask_key].append(sentence_encoding['attention_mask'])
                row_dict[bos_key].append(bos_ids)
            else:
                all_present = False

        if all_present:
            for k , v in row_dict.items():
                encoded_sentences[k] += v

            target_sentence = row[target_key]
            encoded_dict = tokenizer(
                target_sentence,
                max_length=max_length,
                padding="max_length" if pad_to_max_length else None,
                truncation=True,
                return_tensors=return_tensors,
                add_prefix_space = True
            )
            target_ids.append(encoded_dict['input_ids'])

    for key in list(encoded_sentences.keys()):
        encoded_sentences[key] = torch.cat(encoded_sentences[key], dim = 0)
        
    target_ids = torch.cat(target_ids, dim = 0)
    encoded_sentences['labels'] = target_ids
    print(encoded_sentences[key].shape)
    return encoded_sentences

def preprocess_df(df, keys):
    for key in keys:
        df = df[df[key] != "['']"]
    return df

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_files, batch_size, num_examples = 20000 , max_len = 1024, flatten_studies = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.max_len = max_len
        self.flatten_studies = flatten_studies

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train = pd.read_csv(self.data_files[0])
        self.validate = pd.read_csv(self.data_files[1])
        self.test = pd.read_csv(self.data_files[2])
        preprocess_keys = ['population', 'interventions', 'outcomes', 'SummaryConclusions','punchline_text', 'punchline_effect' ]
        self.train = preprocess_df(self.train, preprocess_keys)
        self.validate = preprocess_df(self.validate, preprocess_keys)
        self.test = preprocess_df(self.test, preprocess_keys)

    def setup(self):
        self.train = encode_sentences(self.tokenizer, 
                                      self.train,
                                        ['Abstract'], 
                                        'SummaryConclusions',
                                        max_length = self.max_len)
        self.validate = encode_sentences(self.tokenizer, 
                                        self.validate,
                                        ['Abstract'], 
                                        'SummaryConclusions',
                                        max_length = self.max_len)
        self.test = encode_sentences(self.tokenizer, 
                                    self.test,
                                        ['Abstract'], 
                                        'SummaryConclusions',
                                        max_length = self.max_len)

    def train_dataloader(self, data_type = 'robo'):
        #dataset = TensorDataset
        if data_type == 'robo':
            dataset = TensorDataset(self.train['Abstract_ids'], self.train['Abstract_attention_masks'],
                                    self.train['Abstract_bos_ids'],
                                    self.train['labels'])
                    
                    
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):

        if data_type == 'robo':
            dataset = TensorDataset(self.validate['Abstract_ids'], self.validate['Abstract_attention_masks'],
                                    self.validate['Abstract_bos_ids'],
                                    self.validate['labels'])
        
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data
    
    def test_dataloader(self, data_type = 'robo'):

        if data_type == 'robo':
            dataset = TensorDataset(self.test['Abstract_ids'], self.test['Abstract_attention_masks'],
                                    self.test['Abstract_bos_ids'],
                                    self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data
    



def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/Users/sanjana', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 256):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    print(train_file)
    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = max_len, flatten_studies = True)
    summary_data.prepare_data()
    assert(len(summary_data.train) > 10)
    return summary_data

if __name__ == '__main__':
    additional_special_tokens = [ "<sep>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens)
    #bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']

    
                                    
    
    summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo', path = '/Users/sanjana', files = data_files, max_len = 1024)
    print(summary_data.train)
    summary_data.setup()
    it = summary_data.val_dataloader()
    batches = iter(it)
    batch = next(batches)

    def print_pico(batch):
        population_input_ids = batch[0] if len(batch) >1 else None
        population_attention_masks = batch[1] if len(batch) >1 else None
        print("Abstract")
        print(population_input_ids)
        print(" ".join([tokenizer.decode(w, skip_special_tokens=False, clean_up_tokenization_spaces=True) for w in population_input_ids]))
        print(population_attention_masks)
        print(batch[2])




    for batch in list(batches)[:5]:
        print('||=||' * 100)
        print_pico(batch)

