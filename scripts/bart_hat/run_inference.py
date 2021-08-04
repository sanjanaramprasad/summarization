import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bart.configuration_bart import BartConfig
import torch
import torch.distributed as dist
from torch.nn import functional as F
from BartMultiEncHAT import BartMultiEncHAT
from transformers.generation_utils import GenerationMixin
from run_experiment_hat import LitModel
from transformers import BartTokenizer
import argparse
from rouge import Rouge
from rouge_score import rouge_scorer
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
import pandas as pd
import nltk
from nltk.translate import meteor_score
from DataToGenerator import Data2TextGenerator
import numpy as np
import subprocess, os, sys 
import config_hierarchical_attn 

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})') 

def run_meteor(model_outputs, targets):
    scores = []
    for model_out, tgt in list(zip(model_outputs, targets)):
        met_score = round(meteor_score.meteor_score([tgt], model_out), 4)
        scores.append(met_score)
    return sum(scores)/len(scores)

def run_bleu(model_outputs, targets):
    scores = []
    for model_out, tgt in list(zip(model_outputs, targets)):
        bleu_score = nltk.translate.bleu_score.sentence_bleu([tgt], model_out)
        scores.append(bleu_score)
    return sum(scores)/len(scores)


def run_rouge(model_outputs, targets):
    rougeScores = rouge.get_scores(model_outputs, targets, avg=True)
    return rougeScores

def sample_scorer(sample, model, tokenizer, nbeams, min_len, r_penalty, l_penalty, generator, device):
    targets = []
    model_outputs = []
    rouge = Rouge()
    
    print("Sample scoring")
    for each in sample:
        outputs = generator.generate(each, num_beams = nbeams,  max_length = 400, min_length = min_len, repetition_penalty = r_penalty, length_penalty = l_penalty, encoder_combination_type = 'self_attn', device = device)
        model_output = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in outputs])
        target = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in each[-1]])
        if model_output.strip():
            model_outputs.append(model_output)
            targets.append(target)
            print("MO : ", model_output)
            print('-' * 13)
            print('TARGETS : ', target)
            print('=' * 13)
    print('='*13)
    print("Values: num_beam:%s || min_len:%s || r_penalty:%s || l_penalty:%s"%( nbeams, min_len, r_penalty, l_penalty))
    show_gpu('GPU memory usage after sample:')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    show_gpu('GPU memory usage after clearing cache:')
    rougeScores = rouge.get_scores(model_outputs, targets, avg=True)
    meteorScores = run_meteor(model_outputs, targets)
    bleuScores = run_bleu(model_outputs, targets)
    print("Rouge : ", rougeScores)
    print("Bleu : ", bleuScores)
    print("Meteor : ", meteorScores)
    print('=' * 13)
    return model_outputs, targets,  rougeScores, meteorScores, bleuScores


def parameter_search(sample,  model, tokenizer, device):

        num_beams_list = [3,5]
        min_lengths = [80,90]
        repetition_penalties = [1.0, 2.0]
        length_penalties = [1.0, 2.0]
        final_refs = []
        final_tgts = []
        max_rou1 = 0
        max_roul = 0
        final_num_beam = 0
        final_min_len = 0

        final_rpenalty = 0
        final_lpenalty =0 

        generator = Data2TextGenerator(model, tokenizer)
        for beam in num_beams_list:
                for min_len in min_lengths:
                    for r_penalty in repetition_penalties:
                        for l_penalty in length_penalties:
                            model_outputs, targets,  rougeScores, meteorScores, bleuScores = sample_scorer(sample, model, tokenizer, nbeams = beam, min_len = min_len, r_penalty = r_penalty, l_penalty = l_penalty, generator = generator, device = device)
                            rou1 = rougeScores['rouge-1']['f']
                            roul = rougeScores['rouge-l']['f']
                            if roul > max_roul:
                                final_num_beam = beam
                                final_min_len = min_len
                                final_rpenalty = r_penalty
                                final_lpenalty = l_penalty
                                max_roul = roul
        return final_num_beam, final_min_len, final_rpenalty, final_lpenalty



def run_inference( encoder_combination_strategy, checkpoint_file, parameter_look = True, write_results = True):
    if encoder_combination_strategy == 'HAT':
        config = config_hierarchical_attn

    tokenizer = config.tokenizer
    special_tokens = config.additional_special_tokens
    tokenizer.add_tokens(special_tokens)

    hparams = argparse.Namespace()
    hparams.eval_beams = 4
    device = torch.device("cuda")

    model = config.model
    model.to(device)

    summary_data = config.summary_data
    summary_data.setup("stage")
    val_data = summary_data.val_dataloader()

    num_val = 50
    print("NUM EXAMPLES", num_val)
    it = iter(val_data)
    import random
    sample_data = random.sample(list(it), num_val)


    if parameter_look:
        num_beams, min_len, repetition_penalty, length_penalty = parameter_search(sample_data, model, tokenizer, device)
    
    else:
        num_beams = config.num_beams
        min_len = config.min_len
        repetition_penalty = config.repetition_penalty
        length_penalty = config.length_penalty
    generator = Data2TextGenerator(model, tokenizer)

    model_outputs, targets,  rougeScores, meteorScores, bleuScores = sample_scorer(sample = list(it), model = model, tokenizer = tokenizer, nbeams = num_beams, min_len = min_len, r_penalty = repetition_penalty, l_penalty = length_penalty, generator = generator, device = device) 
    return model_outputs, targets,  rougeScores, meteorScores, bleuScores





if __name__ =='__main__':
    checkpoint_file = ''
    output_file = ''
    encoder_combination_strategy = "addition"
    if not output_file:
        with torch.no_grad():
            model_outputs, targets,  rougeScore, meteorScore, bleuScore = run_inference( encoder_combination_strategy, checkpoint_file)
        df_write = pd.DataFrame(list(zip(targets, model_outputs)), columns=["Reference Summary", "Generated Summary"])
        file_name = "run_inference_output"
        df_write.to_csv("%s.csv"%file_name)


    else:
        df = pd.read_csv(output_file)
        model_outputs = df['Generated Summary']
        targets = df['Reference Summary']
        meteorScore = run_meteor(model_outputs, targets)
        bleuScore = run_bleu(model_outputs, targets)
        rougeScore = run_rouge(model_outputs, targets)
    
    print("Rouge : ", rougeScore)
    print("Bleu : ", bleuScore)
    print("Meteor : ", meteorScore)
    print('=' * 13)
