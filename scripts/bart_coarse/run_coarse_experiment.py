from DataToTextProcessor import SummaryDataModule
import pandas as pd
import transformers
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
#import pandas as pd
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
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.generation_utils import GenerationMixin
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from Coarse_multienc import BartMultiEncCoarse
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pytorch_lightning as pl


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False



def get_data(data):
        population_input_ids = data[0] 
        population_attention_masks = data[1] 
        population_bos_ids = data[2]

        interventions_input_ids = data[3] 
        interventions_attention_masks = data[4] 
        interventions_bos_ids = data[5]


        outcomes_input_ids = data[6] 
        outcomes_attention_masks = data[7] 
        outcomes_bos_ids = data[8]

        punchline_text_input_ids = data[9] 
        punchline_text_attention_masks = data[10] 
        punchline_text_bos_ids = data[11]

        punchline_effect_input_ids = data[12] 
        punchline_effect_attention_masks = data[13] 
        punchline_effect_bos_ids = data[14]


        return population_input_ids, population_attention_masks, population_bos_ids,\
                interventions_input_ids, interventions_attention_masks, interventions_bos_ids,\
                outcomes_input_ids, outcomes_attention_masks, outcomes_bos_ids,\
                punchline_text_input_ids, punchline_text_attention_masks, punchline_text_bos_ids,\
                punchline_effect_input_ids, punchline_effect_attention_masks, punchline_effect_bos_ids


class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, encoder_combination_type, layer_share ,freeze_encoder, freeze_embeds, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model._make_duplicate_encoders(layer_share = layer_share)
        self.model._make_duplicate_decoder_layer_attns()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds_ = freeze_embeds
        self.encoder_combination_type = encoder_combination_type
        self.max_len = max_len

        if self.freeze_encoder:
            freeze_params(self.model.encoder)
            freeze_params(self.model.encoder1)
            freeze_params(self.model.encoder2)
            freeze_params(self.model.encoder3)
            freeze_params(self.model.encoder4)


        if freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.shared)
        for d in [self.model.encoder, self.model.encoder1, self.model.encoder2,
            self.model.encoder3, self.model.encoder4, self.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids_col0, **kwargs):
        return self.model(input_ids_col0, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        optimizer = AdamW(self.model.parameters(),
            lr=3e-4, eps = 1e-7, betas = (0.9, 0.999), weight_decay = 0.01)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = len(self.train_dataloader()) * 12 / num_gpus / 1 / 1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=1300, num_training_steps=num_steps
        )
        #return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        '''optimizer = NoamOpt(768, 2, 4000,
            torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-5))
        #optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr= self.learning_rate)
        return optimize'''

    def training_step(self, batch, batch_idx):
    
        population_input_ids, population_attention_masks, population_bos_ids,\
                interventions_input_ids, interventions_attention_masks, interventions_bos_ids,\
                outcomes_input_ids, outcomes_attention_masks, outcomes_bos_ids,\
                punchline_text_input_ids, punchline_text_attention_masks, punchline_text_bos_ids,\
                punchline_effect_input_ids, punchline_effect_attention_masks, punchline_effect_bos_ids = get_data(batch)

        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        # Run the model and get the logits
        #print(self.encoder_forward_strategy, self.encoder_combination_type)
        outputs = self(
            input_ids_col0 = population_input_ids,
            input_ids_col1 = interventions_input_ids,
            input_ids_col2 = outcomes_input_ids, 
            input_ids_col3 = punchline_text_input_ids,
            input_ids_col4 = punchline_effect_input_ids,
            attention_mask_col0 = population_attention_masks,
            attention_mask_col1 = interventions_attention_masks,
            attention_mask_col2 = outcomes_attention_masks,
            attention_mask_col3 = punchline_text_attention_masks,
            attention_mask_col4 = punchline_effect_attention_masks,
            bos_ids_col0 = population_bos_ids,
            bos_ids_col1 = interventions_bos_ids,
            bos_ids_col2 = outcomes_bos_ids,
            bos_ids_col3 = punchline_text_bos_ids,
            bos_ids_col4 = punchline_effect_bos_ids,
            labels = tgt_ids,
            encoder_combination_type = self.encoder_combination_type,
            decoder_input_ids = None,
            use_cache = False,
        )
        
        loss = outputs[0]
        # Create the loss function
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        tensorboard_logs = {'loss': loss}
        self.log('train_loss', loss)
        epoch_dictionary={

            'loss': loss,
            'log': tensorboard_logs,
            }
        return epoch_dictionary


    def validation_step(self, batch, batch_idx):
        population_input_ids, population_attention_masks, population_bos_ids,\
                interventions_input_ids, interventions_attention_masks, interventions_bos_ids,\
                outcomes_input_ids, outcomes_attention_masks, outcomes_bos_ids,\
                punchline_text_input_ids, punchline_text_attention_masks, punchline_text_bos_ids,\
                punchline_effect_input_ids, punchline_effect_attention_masks, punchline_effect_bos_ids = get_data(batch)

        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        # Run the model and get the logits
        #print(self.encoder_forward_strategy, self.encoder_combination_type)
        outputs = self(
            input_ids_col0 = population_input_ids,
            input_ids_col1 = interventions_input_ids,
            input_ids_col2 = outcomes_input_ids, 
            input_ids_col3 = punchline_text_input_ids,
            input_ids_col4 = punchline_effect_input_ids,
            attention_mask_col0 = population_attention_masks,
            attention_mask_col1 = interventions_attention_masks,
            attention_mask_col2 = outcomes_attention_masks,
            attention_mask_col3 = punchline_text_attention_masks,
            attention_mask_col4 = punchline_effect_attention_masks,
            bos_ids_col0 = population_bos_ids,
            bos_ids_col1 = interventions_bos_ids,
            bos_ids_col2 = outcomes_bos_ids,
            bos_ids_col3 = punchline_text_bos_ids,
            bos_ids_col4 = punchline_effect_bos_ids,
            labels = tgt_ids,
            encoder_combination_type = self.encoder_combination_type,
            decoder_input_ids = None,
            use_cache = False,
        )

        val_loss = outputs[0]

        tensorboard_logs = {'val_loss': val_loss}
        self.log('val_loss_epoch', val_loss)
        epoch_dictionary={
            'val_loss': val_loss,
            'log': tensorboard_logs}
        return epoch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}



def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 1024):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])



    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = max_len)
    summary_data.prepare_data()
    assert(len(summary_data.train) > 10)
    return summary_data




def main(encoder_combination_type = 'HAT', layer_share = False):
    #additional_special_tokens=["<attribute>",  "</attribute>", "<sep>"]
    #

    ############################# Data loader and data prep ####################
    additional_special_tokens = ["<sep>"]
    '''additional_special_tokens = ['<population>', '</population>',
                                        '<interventions>', '</interventions>',
                                        '<outcomes>', '</outcomes>',
                                        '<punchline_text>', '</punchline_text>',
                                        '<punchline_effect>', '</punchline_effect>', "<sep>"]'''

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens) 
    ##from Data2TextProcessor import SummaryDataModule
    files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv']
    max_len = 1024

    
    summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/ramprasad.sa', files = files, max_len = max_len)





    ####################### Model loading and training ##########################
    freeze_encoder = False
    freeze_embeds = False
    learning_rate = 3e-5 
    max_epochs = 20
    bart_model = BartMultiEncCoarse.from_pretrained('facebook/bart-base') 
    logger = TensorBoardLogger('tb_logs_final', name='my_model_f%s'%(encoder_combination_type))  
    model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = bart_model, 
                    encoder_combination_type = encoder_combination_type, layer_share = layer_share, freeze_encoder = freeze_encoder, \
                            freeze_embeds = freeze_embeds, max_len = max_len)

    checkpoint = ModelCheckpoint('checkpoint_files/3e-5_encoder_coarse',
                                filename = '{epoch}-{val_loss:.2f}',
                                save_top_k=10,
                                monitor = 'val_loss')
    trainer = pl.Trainer(gpus=1, accelerator='dp', 
			max_epochs = max_epochs,
                        min_epochs = 1,
                        auto_lr_find = False,
                        progress_bar_refresh_rate = 100,
                        logger=logger,
                        callbacks=[checkpoint])

    trainer.fit(model, summary_data)















if __name__ == '__main__': 
    main(encoder_combination_type = 'coarse_attention')
