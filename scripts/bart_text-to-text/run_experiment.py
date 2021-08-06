import pandas as pd 
from DataToTextProcessor import SummaryDataModule 
import transformers
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
##import pandas as pd
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
##from BartForDataToTextGeneration import BartForDataToText
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
## from Data2TextProcessor_linearize import SummaryDataModule
#from transformers.modeling_bart import shift_tokens_right
from rouge import Rouge
import nltk
from nltk.translate import meteor_score

##import pandas as pd
learning_rate = 3e-5 
max_epochs = 13

logger = TensorBoardLogger('tb_logs', name='my_model_epoch_%s_%s'%(str(max_epochs), str(learning_rate)))


train_count = 0
val_count = 0

import os
import pytorch_lightning as pl



    
def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False


class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, freeze_encoder, freeze_embeds, eval_beams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds_ = freeze_embeds
        #self.hparams = hparams
        #self.hparams.update(hparams)

        if self.freeze_encoder:
            freeze_params(self.model.get_encoder())


        if freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr= self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        #print(batch)
    
        input_ids = batch[0] 
        attention_mask= batch[1]
        bos_ids = batch[2]
        tgt_ids = batch[-1]
        outputs = self(
            input_ids= input_ids,
            attention_mask= attention_mask,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
        )
        
        #lm_logits = outputs[1]
        # Create the loss function
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        loss = outputs[0]
        tensorboard_logs = {'loss': loss}
        self.log("train_loss", loss)
        epoch_dictionary={
            'loss': loss,
            'log': tensorboard_logs}
        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
    
        
        input_ids = batch[0] 
        attention_mask= batch[1]
        bos_ids = batch[2]
        tgt_ids = batch[-1]
        outputs = self(
            input_ids= input_ids,
            attention_mask= attention_mask,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
        )
        
        #lm_logits = outputs[1]
        # Create the loss function
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        loss = outputs[0]
        tensorboard_logs = {'loss': loss}
        self.log("val_loss_step", loss)

        epoch_dictionary={
            'val_loss': loss,
            'log': tensorboard_logs}
        return epoch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log("val_loss", avg_loss)

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


def main():
    additional_special_tokens = ["<sep>"]

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens) 
    ##from Data2TextProcessor import SummaryDataModule
    files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv']
    max_len = 1024

    
    summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/ramprasad.sa', files = files, max_len = max_len)
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    bart_model.resize_token_embeddings(len(tokenizer))
    #hparams = argparse.Namespace()
    freeze_encoder = False
    freeze_embeds = False
    eval_beams = 4

    model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = bart_model, freeze_encoder = freeze_encoder, freeze_embeds = freeze_embeds, eval_beams = eval_beams)
    checkpoint = ModelCheckpoint('checkpoint_files/',
                                filename = '{epoch}-{val_loss:.2f}',
                                save_top_k=13,
                                monitor = 'val_loss')
    trainer = pl.Trainer(gpus=1, accelerator='dp', 
			max_epochs = max_epochs,
                        min_epochs = 1,
                        auto_lr_find = False,
                        progress_bar_refresh_rate = 100,
                        logger=logger,
                        callbacks=[checkpoint])

    trainer.fit(model, summary_data)

def inference(checkpoint_file):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    hparams = argparse.Namespace()
    rouge = Rouge()
    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4
    model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_file)

    summary_data = make_data(tokenizer, path = '/home/sanjana')
    summary_data.setup("stage")
    val_data = summary_data.val_dataloader(data_type = 'robo')

    num_val = len(list(val_data))
    num_val = 5
    print("NUM EXAMPLES", num_val)
    it = iter(val_data)
    ind = 0
    model_out = []
    references = []
    rouge = Rouge()
    meteor_scores = []
    bleu_scores =[]
    
    for text in it:
        generated_ids = model.model.generate(
                text[0],
                attention_mask=text[1],
                use_cache=True,
                decoder_start_token_id = tokenizer.pad_token_id,
                num_beams= 3,
                min_length = 70,
                max_length = 300,
                early_stopping = True
        )
    
        model_output = " ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids])
        target = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in text[-1]])
        print(target, model_output)
        references.append(target)
        model_out.append(model_output)
        met_score = round(meteor_score.meteor_score([target], model_output), 4)
        meteor_scores.append(met_score)
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], model_output)
        bleu_scores.append(BLEUscore)
    print("ROGUE", rouge.get_scores(model_out, references, avg=True))
    print("METEOR", sum(meteor_scores)/len(meteor_scores))
    print("BLEU", sum(bleu_scores)/len(bleu_scores))
if __name__ == '__main__': 
    main()
    #inference('/home/sanjana/roboreviewer_summarization/scripts/bart_vanilla/checkpoint_files/checkpoint_best_model/epoch=4-val_loss=0.25.ckpt')
   
