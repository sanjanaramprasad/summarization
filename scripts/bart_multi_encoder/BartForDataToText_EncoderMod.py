import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartPretrainedModel, shift_tokens_right, BartDecoderLayer, BartLearnedPositionalEmbedding, BartAttention, _make_causal_mask, _expand_mask

from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput,Seq2SeqSequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import copy
from transformers.activations import ACT2FN


class BartEncoderShared():
    def __init__(self, enc, layers, shared_num ):
        ind = 0
        own_layers = enc.layers[shared_num:]
        for shared_layer in layers:
            own_layers.insert(ind, shared_layer)
            ind +=1
        enc.layers = own_layers


class BartForDataToText(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    
    def __init__(self, config: BartConfig):
        super().__init__(config)
        print(config.encoder_layers)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout


        self.encoder = BartEncoder(config, self.shared)
        

        
        self.decoder = BartDecoder(config,self.shared)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        self.fc0 = nn.Linear(config.d_model * 5, config.d_model * 2)
        self.final_layer = nn.Linear(config.d_model * 2, config.d_model)

        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        #print("DIM", config.d_model)
        self.init_weights()
        

        
    def _make_duplicate_encoders(self, layer_share = False):
        self.encoder1 = copy.deepcopy(self.encoder)
        self.encoder2 = copy.deepcopy(self.encoder)
        self.encoder3 = copy.deepcopy(self.encoder)
        self.encoder4 = copy.deepcopy(self.encoder)
        if layer_share:
            BartEncoderShared(self.encoder1, self.encoder.layers[:3], 3)
            BartEncoderShared(self.encoder2, self.encoder.layers[:3], 3)
            BartEncoderShared(self.encoder3, self.encoder.layers[:3], 3)
            BartEncoderShared(self.encoder4, self.encoder.layers[:3], 3)
        
    def get_input_embeddings(self):
        return self.shared 
    
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = value
        self.encoder1.embed_tokens = value
        self.encoder2.embed_tokens = value
        self.encoder3.embed_tokens = value
        self.encoder4.embed_tokens = value
        self.decoder.embed_tokens = value
        
    def get_encoders(self):
        return self.encoder, self.encoder1, \
            self.encoder2, self.encoder3, self.encoder4
    
    def get_decoder(self):
        self.decoder
        
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings
    
    def _get_encoder_outputs(self, 
            encoder , 
            encoder_outputs, 
            input_ids,
            attention_mask,
            head_mask = None,
            inputs_embeds = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict= None):
        
        if encoder_outputs is None:
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        return encoder_outputs
        
    
    def _get_sum_encoder_outputs(self,
            encoder_output_list, average_flag):
        encoder_outputs = {0:[], 1:[], 2:[]}
        for i in range(0,3):
            if len(encoder_output_list[0]) > i:
                added_enc_outputs_i = torch.stack([enc[i] for enc in encoder_output_list], dim = 0)
                if average_flag:
                    added_enc_outputs_i = torch.mean(added_enc_outputs_i, dim = 0)
                else:
                    added_enc_outputs_i = torch.sum(added_enc_outputs_i, dim = 0)

                encoder_outputs[i].append(added_enc_outputs_i)

        added_enc_outputs = BaseModelOutput(
                last_hidden_state=torch.cat(encoder_outputs[0], dim =0 ),
                hidden_states=torch.cat(encoder_outputs[1], dim =0 ) if len(encoder_outputs[1]) > 0 else None,
                attentions=torch.cat(encoder_outputs[2], dim =0 ) if len(encoder_outputs[2]) > 0 else None,
            )
        #print(added_enc_outputs)
        return added_enc_outputs


    def _get_concat_encoder_outputs(self, 
        encoder_outputs_list):

        encoder_outputs = {0:[], 1:[], 2:[]}
        for i in range(0,3):
            if len(encoder_outputs_list[0]) > i: 
                added_enc_outputs_i = torch.cat([enc[i] for enc in encoder_outputs_list],2)
                encoder_outputs[i].append(added_enc_outputs_i)
            
        added_enc_outputs = BaseModelOutput(
                last_hidden_state=torch.cat(encoder_outputs[0], dim =0 ),
                hidden_states=torch.cat(encoder_outputs[1], dim =0 ) if len(encoder_outputs[1]) > 0 else None,
                attentions=torch.cat(encoder_outputs[2], dim =0 ) if len(encoder_outputs[2]) > 0 else None,
            )
        #print(added_enc_outputs)
        return added_enc_outputs
        
    
    def _get_attention_masks_OR(self, 
        attention_mask_list ):

            #all_attn_outputs = torch.cat(attention_mask_list, 1)

            all_attn_outputs = torch.stack(attention_mask_list, 0)
            added_enc_attns = torch.Tensor.float(all_attn_outputs).mean(0).tolist()
            added_enc_attns = [[1 if each > 0 else 0 for each in each_list] for each_list in added_enc_attns]
            #added_enc_attns = torch.as_tensor([added_enc_attns])
            added_enc_attns = torch.as_tensor(added_enc_attns , device = attention_mask_list[0].device)
            return added_enc_attns
    
    def _forward_pass(self, encoder_outputs, fc0 , fc1, final_layer):
        enc_outputs = []
        for i in range(0,3):
            if len(encoder_outputs) > i:
                #fcn(encoder_outputs[i]))
                layer_1 = self.activation_fn(fc0(encoder_outputs[i]))
                layer_1 = F.dropout(layer_1, p=self.activation_dropout, training=self.training)
                layer_2 = final_layer(layer_1)
                layer_2 = F.dropout(layer_2,  p=self.dropout, training=self.training)
                enc_outputs.append(layer_2)


        encoder_outputs = BaseModelOutput(
                last_hidden_state=enc_outputs[0],
                hidden_states=enc_outputs[1] if len(enc_outputs) > 1 else None,
                attentions=enc_outputs[2] if len(enc_outputs) > 2 else None,
            )
        return encoder_outputs


    def forward(
        self,
        input_ids_col0 = None,
        input_ids_col1 = None,
        input_ids_col2 = None, 
        input_ids_col3 = None,
        input_ids_col4 = None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        attention_mask_col4 = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_col0 = None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_combination_type = 'addition',
        check_status = False
    ):
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        attn_mask_list = []
        encoder_outputs_list =[]

        if not (input_ids_col0 is None):
                encoder_outputs_col0 = self._get_encoder_outputs(
                            encoder = self.encoder, 
                            encoder_outputs = encoder_outputs_col0, 
                            input_ids = input_ids_col0,
                            attention_mask = attention_mask_col0,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            return_dict = return_dict)
                encoder_outputs_list.append(encoder_outputs_col0)
                attn_mask_list.append(attention_mask_col0)

        if not (input_ids_col1 is None):
                encoder_outputs_col1 = self._get_encoder_outputs(
                            encoder = self.encoder1, 
                            encoder_outputs = encoder_outputs_col1, 
                            input_ids = input_ids_col1,
                            attention_mask = attention_mask_col1,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            return_dict = return_dict)
                encoder_outputs_list.append(encoder_outputs_col1)
                attn_mask_list.append(attention_mask_col1)

        if not (input_ids_col2 is None):
                encoder_outputs_col2 = self._get_encoder_outputs(
                            encoder = self.encoder2, 
                            encoder_outputs = encoder_outputs_col2, 
                            input_ids = input_ids_col2,
                            attention_mask = attention_mask_col2,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            return_dict = return_dict)
                encoder_outputs_list.append(encoder_outputs_col2)
                attn_mask_list.append(attention_mask_col2)
        
        if not (input_ids_col3 is None):
                encoder_outputs_col3 = self._get_encoder_outputs(
                            encoder = self.encoder3, 
                            encoder_outputs = encoder_outputs_col3, 
                            input_ids = input_ids_col3,
                            attention_mask = attention_mask_col3,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            return_dict = return_dict)
                encoder_outputs_list.append(encoder_outputs_col3)
                attn_mask_list.append(attention_mask_col3)
        
        if not (input_ids_col4 is None):
                encoder_outputs_col4 = self._get_encoder_outputs(
                            encoder = self.encoder4, 
                            encoder_outputs = encoder_outputs_col4, 
                            input_ids = input_ids_col4,
                            attention_mask = attention_mask_col4,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            return_dict = return_dict)
                encoder_outputs_list.append(encoder_outputs_col4)
                attn_mask_list.append(attention_mask_col4)
            

        
        
       #print(encoder_outputs_list)
        encoder_outputs = [encoder_outputs_col0, encoder_outputs_col1, encoder_outputs_col2, encoder_outputs_col3, encoder_outputs_col4]
        encoder_outputs_list = [each for each in encoder_outputs if each is not None]
        attention_masks = [attention_mask_col0, attention_mask_col1, attention_mask_col2, attention_mask_col3, attention_mask_col4]
        attn_mask_list = [each for each in attention_masks if each is not None]


        if len(encoder_outputs_list) == 1:
            encoder_outputs = encoder_outputs_list[0]
            attn_mask = attn_mask_list[0] if attn_mask_list else None

        else:
    
            if encoder_combination_type =='addition':
                average_flag = False
                encoder_outputs = self._get_sum_encoder_outputs(
                        encoder_outputs_list,
                        average_flag
                    )
                
                if check_status:
                    encoder_sum_outputs_0 = encoder_outputs_col0[0] + encoder_outputs_col1[0] + encoder_outputs_col2[0] + \
                            encoder_outputs_col3[0] + encoder_outputs_col4[0]
                    if average_flag:
                        encoder_sum_outputs_0 = encoder_sum_outputs_0 / 5 
                    encoder_outputs[0] == encoder_sum_outputs_0
                    assert(bool(encoder_sum_outputs_0.all()))
                    

                if attention_mask_col0 is None:
                    attn_mask = attention_mask_col0
                else:
                    attn_mask = self._get_attention_masks_OR(
                        [attn_mask for attn_mask in attn_mask_list if not (attn_mask is None)]

                    )
            elif encoder_combination_type == 'linearize':
                encoder_outputs = self._get_concat_encoder_outputs(encoder_outputs_list)
                encoder_outputs = self._forward_pass(encoder_outputs, self.fc0, self.fc1, self.final_layer)
                attn_mask = None


            elif encoder_combination_type == 'self_attn':
                encoder_outputs = self._get_concat_encoder_outputs(encoder_outputs_list)
                
                encoder_outputs_concat, concat_attn_weights, _ = self.encoder_attn(
                hidden_states = encoder_outputs,
                attention_mask = decoder_head_mask,
                layer_head_mask=None,
                output_attentions=output_attentions,
                )
                encoder_outputs = self._forward_pass(encoder_outputs, self.fc0, self.fc1, self.final_layer)
                attn_mask = None


        
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attn_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            outputs =  decoder_outputs + encoder_outputs
            
        else:
            outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
            
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        attention_mask_col4 = None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_col0 =None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        encoder_forward_strategy = 'single',
        encoder_combination_type = 'addition',
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids_col0": None,
            "input_ids_col1": None,
            "input_ids_col2": None,
            "input_ids_col3": None,
            "input_ids_col4": None,
            "encoder_outputs_col0": encoder_outputs_col0,
            "encoder_outputs_col1": encoder_outputs_col1,
            "encoder_outputs_col2": encoder_outputs_col2,
            "encoder_outputs_col3": encoder_outputs_col3,
            "encoder_outputs_col4": encoder_outputs_col4,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask_col0": attention_mask_col0,
            "attention_mask_col1": attention_mask_col1,
            "attention_mask_col2": attention_mask_col2,
            "attention_mask_col3": attention_mask_col3,
            "attention_mask_col4": attention_mask_col4,
            "encoder_forward_strategy": encoder_forward_strategy,
            "encoder_combination_type": encoder_combination_type,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            
        }
        
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past        
    
        



