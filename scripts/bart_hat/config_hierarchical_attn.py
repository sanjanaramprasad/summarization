#from transformers import BartTokenizer
from DataToTextProcessor import SummaryDataModule
from transformers import BartTokenizer
import subprocess, os, sys 
from run_experiment_hat import LitModel


parent_dir_name = "home/sanjana"

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


additional_special_tokens = ["<sep>", "[BOS]"]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

tokenizer.add_tokens(additional_special_tokens) 
files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv']
max_len = 1024


summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/sanjana', files = files, max_len = max_len)
print(summary_data.train)
checkpoint_file = '/home/sanjana/summarization/scripts/bart_hat/trial/epoch=7-val_loss=0.54.ckpt'

model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_file)

num_beams = 3
min_len = 90
repetition_penalty = 2.0
length_penalty = 2.0
    
