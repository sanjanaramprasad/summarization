
import transformers
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_utils import GenerationMixin
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from DataToTextProcessor import SummaryDataModule




class BartForConditionalGenerationTester():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        bos_ids = batch[2]
        tgt_ids = batch[-1]

        outputs = self.model(input_ids = input_ids,
                attention_mask = attention_mask,
                labels = tgt_ids,
                decoder_input_ids = None,
                use_cache = False)
        optimizer = optim.Adam(self.model.parameters())
        lm_logits = outputs[1]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()

        return outputs, loss 


def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/sanjana', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 1024):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])



    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = max_len)
    summary_data.prepare_data()
    assert(len(summary_data.train) > 10)
    return summary_data





print("Init tokenizer ...")
additional_special_tokens = ["<sep>"]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

tokenizer.add_tokens(additional_special_tokens)
files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv']
max_len = 1024                                       
summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/sanjana', files = files, max_len = max_len)
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
bart_model.resize_token_embeddings(len(tokenizer))
#hparams = argparse.Namespace()
freeze_encoder = False
freeze_embeds = False
eval_beams = 4
obj = BartForConditionalGenerationTester(bart_model, tokenizer)
#bart_model.resize_token_embeddings(len(self.tokenizer))
print("Making data")
summary_data = make_data(tokenizer, path = '/home/sanjana')
summary_data.prepare_data()
summary_data.setup("stage")
train_data = summary_data.train_dataloader(data_type = 'robo')
it = iter(train_data)
batch = next(it)
print(obj.forward(batch))

