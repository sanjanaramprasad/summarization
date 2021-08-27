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


def preprocess_df(df, keys):
    for key in keys:
        df = df[df[key] != "['']"]
    return df

def encode_sentences(tokenizer, df, source_keys, targets, max_length=1024, pad_to_max_length=True, return_tensors="pt"):

    input_ids = []
    attention_masks = []
    target_ids = []


    for idx, row in df.iterrows():
        row_sentence = []
        all_sents = []
        row_sentence = eval(row['Abstract'])
        for v in row_sentence:
            all_sents.append("<study> " + v + " </study>")
        sentence = " ".join(all_sents)
        ##print(sentence)
        encoded_dict = tokenizer(
          sentence,
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
        
    
    for tgt_sentence in targets:
        print(tgt_sentence)
        encoded_dict = tokenizer(
              tgt_sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
        )
        # Shift the target ids to the right
        #shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])
    
    target_ids = torch.cat(target_ids, dim = 0)
        
    encode_sentences  = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return encode_sentences

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

    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, 
                                      self.train,
                                        ['Abstract'], 
                                        self.train['SummaryConclusions'],
                                        max_length = self.max_len)
        self.validate = encode_sentences(self.tokenizer, 
                                        self.validate,
                                        ['Abstract'], 
                                        self.validate['SummaryConclusions'],
                                        max_length = self.max_len)
        self.test = encode_sentences(self.tokenizer, 
                                    self.test,
                                    ['Abstract'], 
                                    self.test['SummaryConclusions'],
                                    max_length = self.max_len)

    def train_dataloader(self, data_type = 'robo'):
        #dataset = TensorDataset
        if data_type == 'robo':
            dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'],
                                    self.train['labels'])
                    
                    
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):

        if data_type == 'robo':
            dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'],
                                    self.validate['labels'])
        
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data
    
    def test_dataloader(self, data_type = 'robo'):

        if data_type == 'robo':
            dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'],
                                    self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data
    



def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 256):
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
    summary_data.setup("stage")
    it = summary_data.val_dataloader()
    batches = iter(it)
    batch = next(batches)

    def print_pico(batch):
        population_input_ids = batch[0] if len(batch) >1 else None
        population_attention_masks = batch[1] if len(batch) >1 else None
        print("Abstract")
        print(population_input_ids)
        print(" ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in population_input_ids]))
        print(population_attention_masks)
        print(batch[2])




    for batch in list(batches)[:5]:
        print('||=||' * 100)
        print_pico(batch)

