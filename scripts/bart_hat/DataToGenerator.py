
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bart.configuration_bart import BartConfig
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers.generation_utils import GenerationMixin
from transformers import BartTokenizer
import argparse
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
import pandas as pd
import nltk
from nltk.translate import meteor_score
import numpy as np
import subprocess 


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

class Data2TextGenerator(GenerationMixin):

    def __init__(self, model, tokenizer):
        self.model = model.model 
        self.tokenizer = tokenizer 
        self.config = self.model.config
        self.device = self.model.device
        #print(self.config.max_length)



    def _prepare_attention_mask_for_generation(self, batch, device, model_kwargs):
        attention_mask_col0 = batch[1] 
        attention_mask_col1 = batch[4] 
        attention_mask_col2 = batch[7] 
        attention_mask_col3 = batch[10] 
        attention_mask_col4 = batch[13] 
        
        if not(attention_mask_col0 is None):
            model_kwargs["attention_mask_col0"] = attention_mask_col0
            model_kwargs["attention_mask_col0"] = model_kwargs["attention_mask_col0"].to(device)

        if not(attention_mask_col1 is None):
            model_kwargs["attention_mask_col1"] = attention_mask_col1
            model_kwargs["attention_mask_col1"] = model_kwargs["attention_mask_col1"].to(device)

        if not(attention_mask_col2 is None):
            model_kwargs["attention_mask_col2"] = attention_mask_col2
            model_kwargs["attention_mask_col2"] = model_kwargs["attention_mask_col2"].to(device)

        if not(attention_mask_col3 is None):
            model_kwargs["attention_mask_col3"] = attention_mask_col3
            model_kwargs["attention_mask_col3"] = model_kwargs["attention_mask_col3"].to(device)

        if not(attention_mask_col4 is None):
            model_kwargs["attention_mask_col4"] = attention_mask_col4
            model_kwargs["attention_mask_col4"] = model_kwargs["attention_mask_col4"].to(device)

        #print(model_kwargs)
        return model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, 
        population_input_ids, 
        interventions_input_ids,
        outcomes_input_ids,
        punchline_text_input_ids,
        punchline_effect_input_ids,
        population_bos_ids,
        interventions_bos_ids,
        outcomes_bos_ids,
        punchline_text_bos_ids,
        punchline_effect_bos_ids, 
        device,
        model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder_col0, encoder_col1, encoder_col2, encoder_col3, encoder_col4 = self.model.get_encoders()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            if not(population_input_ids is None):
                encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                attention_mask_col0 = encoder_kwargs.get("attention_mask_col0", None)
                encoder_outputs = encoder_kwargs.get('encoder_outputs_col0', None)
                
                model_kwargs["encoder_outputs_col0"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col0, encoder_outputs = encoder_outputs, input_ids = population_input_ids, attention_mask = attention_mask_col0)
                
            if not(interventions_input_ids is None):
                    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                    attention_mask_col1 = encoder_kwargs.get("attention_mask_col1", None)
                    encoder_outputs = encoder_kwargs.get('encoder_outputs_col1', None)

                    model_kwargs["encoder_outputs_col1"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col1, encoder_outputs = encoder_outputs, input_ids = interventions_input_ids, attention_mask = attention_mask_col1)
                    
            if not(outcomes_input_ids is None):
                    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                    attention_mask_col2 = encoder_kwargs.get("attention_mask_col2", None)
                    encoder_outputs = encoder_kwargs.get('encoder_outputs_col2', None)

                    model_kwargs["encoder_outputs_col2"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col2, encoder_outputs = encoder_outputs, input_ids = outcomes_input_ids, attention_mask = attention_mask_col2)
                    
            if not(punchline_text_input_ids is None):
                    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                    attention_mask_col3 = encoder_kwargs.get("attention_mask_col3", None)
                    encoder_outputs = encoder_kwargs.get('encoder_outputs_col3', None)
                    
                    model_kwargs["encoder_outputs_col3"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col3, encoder_outputs = encoder_outputs, input_ids = punchline_text_input_ids, attention_mask = attention_mask_col3)
                    
            if not(punchline_effect_input_ids is None):
                    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not "col" in argument}
                    attention_mask_col4 = encoder_kwargs.get("attention_mask_col4", None)
                    encoder_outputs = encoder_kwargs.get('encoder_outputs_col4', None)

                    model_kwargs["encoder_outputs_col4"]: ModelOutput = self.model._get_encoder_outputs(encoder = encoder_col4, encoder_outputs = encoder_outputs, input_ids = punchline_effect_input_ids, attention_mask = attention_mask_col4)
                    
                    #model_kwargs["encoder_outputs_col4"] = model_kwargs["encoder_outputs_col4"].to(device)

            model_kwargs["bos_ids_col0"] = population_bos_ids
            model_kwargs["bos_ids_col1"] = interventions_bos_ids
            model_kwargs["bos_ids_col2"] = outcomes_bos_ids
            model_kwargs["bos_ids_col3"] = punchline_text_bos_ids
            model_kwargs["bos_ids_col4"] = punchline_effect_bos_ids

        return model_kwargs
    
    def generate(self,
        batch,
        input_ids = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        device = torch.device('cuda'),
        **model_kwargs, 
    ):

        population_input_ids, population_attention_masks, population_bos_ids,\
                interventions_input_ids, interventions_attention_masks, interventions_bos_ids,\
                outcomes_input_ids, outcomes_attention_masks, outcomes_bos_ids,\
                punchline_text_input_ids, punchline_text_attention_masks, punchline_text_bos_ids,\
                punchline_effect_input_ids, punchline_effect_attention_masks, punchline_effect_bos_ids = get_data(data)

        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs_col0"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs =  self._prepare_attention_mask_for_generation(
                batch, device, model_kwargs)

        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None
        input_list = [each for each in [input_ids_col0, input_ids_col1, input_ids_col2, input_ids_col3, input_ids_col4,] \
                            if not(each is None)]
        encoder_input_ids = torch.cat(input_list, 0)
        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(population_input_ids, 
                                                                                interventions_input_ids,
                                                                                outcomes_input_ids,
                                                                                punchline_text_input_ids,
                                                                                punchline_effect_input_ids,
                                                                                population_bos_ids,
                                                                                interventions_bos_ids,
                                                                                outcomes_bos_ids, 
                                                                                punchline_text_bos_ids,
                                                                                punchline_effect_bos_ids,
                                                                                device,
                                                                                model_kwargs,
                                                                                )

            if "decoder_input_ids" in model_kwargs:
                    input_ids = model_kwargs.pop("decoder_input_ids")
                else:
                    input_ids = self._prepare_decoder_input_ids_for_generation(
                        input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                    )

        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        stopping_criteria = self._get_stopping_criteria(max_length=max_length, max_time=max_time)
        is_greedy_gen_mode = True
        if is_greedy_gen_mode:
            #print("GREEDY SEARCHING")
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.model.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        
        