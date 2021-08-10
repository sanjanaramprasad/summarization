import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartPretrainedModel, shift_tokens_right, BartDecoderLayer, BartLearnedPositionalEmbedding, BartAttention, _make_causal_mask, _expand_mask
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput,Seq2SeqSequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
import copy
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN
import random
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)




class BartDecoderLayerMulti(nn.Module):

    def __init__(self, config : BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.hierarchical_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        #self.self_attn2_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.hierarchical_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        sentence_hidden_states : Optional[torch.Tensor] = None,
        sentence_attention_mask: Optional[torch.Tensor] = None,

        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
	    decoder_combination = 'addition'
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        #use_cache = False
        residual = hidden_states
        #if past_key_value:
        #print("PS Value", len(past_key_value))
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
            ####HIERARCH ATTN BLOCK
            ##print("HIERARCH")
            residual = hidden_states
            hierarchical_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states_all, hierarchical_attn_weights, hierarchical_attn_present_key_value = self.hierarchical_attn(
                                                                                                        hidden_states=hidden_states,
                                                                                                        key_value_states=sentence_hidden_states,
                                                                                                        attention_mask=sentence_attention_mask,
                                                                                                        layer_head_mask=cross_attn_layer_head_mask,
                                                                                                        past_key_value=hierarchical_attn_past_key_value,
                                                                                                        output_attentions=output_attentions,
                                                                                                    )
            hidden_states = hidden_states_all + residual
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = self.hierarchical_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + hierarchical_attn_present_key_value
            ######################
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_past_key_value)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartDecoderMulti(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayerMulti(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        sentence_hidden_states = None,
        sentence_attention_mask = None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        decoder_combination = 'addition',
        return_dict=None,
    ):

        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.
                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if sentence_hidden_states is not None and sentence_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            sentence_attention_mask = _expand_mask(sentence_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    sentence_hidden_states = sentence_hidden_states,
                    sentence_attention_mask = sentence_attention_mask,

                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    decoder_combination = decoder_combination,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)


        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class BartEncoderShared():
    def __init__(self, enc, layers, shared_num ):
        ind = 0
        own_layers = enc.layers[shared_num:]
        for shared_layer in layers:
            own_layers.insert(ind, shared_layer)
            ind +=1
        enc.layers = own_layers




class BartMultiEncFlatHAT(BartPretrainedModel):
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

        enc_concat_dim = 256 * 13

        self.encoder = BartEncoder(config, self.shared)

        self.encoder_attn_concat = BartAttention(
            config.d_model * 5,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.encoder_attn_concat_norm = nn.LayerNorm(config.d_model * 5)

        
        self.hierarchical_attention = BartAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.hierarchical_attention_layer_norm = nn.LayerNorm(config.d_model)

        self.concat_proj = nn.Linear(config.d_model*5, config.d_model)
        self.fc1_ha = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2_ha = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        self.decoder = BartDecoderMulti(config,self.shared)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
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

    def _make_duplicate_decoder_layer_attns(self):
        for each_layer in self.decoder.layers:
            each_layer._make_duplicate_attns()

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

    def hierarchical_attn_forward(self, hidden_states, attention_mask, layer_head_mask = None, output_attentions = False):

        residual = hidden_states
        attention_mask = _expand_mask(attention_mask, torch.float32, hidden_states.shape[1])
        hidden_states, attn_weights, _ = self.hierarchical_attention(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            layer_head_mask = layer_head_mask,
            output_attentions = output_attentions
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.hierarchical_attention_layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=None, attentions=None
        )

    def _get_self_attn(self, encoder_outputs_concat, attention_mask, encoder_outputs_list = None):
        residual = encoder_outputs_concat[0]

        ##### Self Attention Block ####
        attn_encoder_outputs_concat, _, _ = self.encoder_attn_concat(
                hidden_states = encoder_outputs_concat[0],
                attention_mask = attention_mask,
                layer_head_mask=None,
                output_attentions=False,
                )

        encoder_contexts = torch.chunk(attn_encoder_outputs_concat, 5, dim = 2)
        all_encoder_vectors = []

        for i, enc_i in enumerate(encoder_outputs_list):
            enc_i = enc_i[0]
            encoder_vectors_diff = []
            encoder_vectors_mult = []
            context_i = encoder_contexts[i]
            print("ENCODER NUM : ", i)
            for j in range(0, len(encoder_contexts)):
                context_j = encoder_contexts[j]
                if i != j:
                    diff = torch.sub(enc_i, context_j)
                    mult = torch.mul(enc_i, context_j)
                    encoder_vectors_diff.append(diff)
                    encoder_vectors_mult.append(mult)
                    print("CONTEXTS : ", j, )
            print('=' * 13)
            
            e_vect = [enc_i] + encoder_vectors_diff + encoder_vectors_mult
            #e_vect = torch.cat(e_vect,2)
            print(len(e_vect), torch.cat(e_vect, dim = 2).shape)
            print('***' * 13)
            all_encoder_vectors += e_vect
        
        all_encoder_vectors = torch.cat(all_encoder_vectors, dim = 2)
        


        #print("ALL INFO SHAPE", all_encoder_vectors.shape)

        #attn_encoder_outputs_concat = F.dropout(attn_encoder_outputs_concat, p=self.dropout, training=self.training)

        #hidden_states = attn_encoder_outputs_concat + residual
        #hidden_states = self.encoder_attn_concat_norm(hidden_states)
        #print("HIDDEN STATES", hidden_states.shape)
        hidden_states = self.concat_proj(hidden_states)

        encoder_outputs = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=None,
                attentions=None,
                )
        return encoder_outputs


    
    def _get_sentence_vectors(self, encoder_output_list, bos_id_list):
        vector_list = []
        vector_attention = []
        max_len = encoder_output_list[0][0][0].shape[0]
        embed_dim = encoder_output_list[0][0][0].shape[1]

        #print("MAX LEN", max_len)
        #print("EMB DIM", embed_dim)

        for enc_output, bos_ids in list(zip(encoder_output_list, bos_id_list)):
            enc_last_hidden_state = enc_output[0]
            enc_last_hs_vectors = enc_last_hidden_state[0]
            #sentence_output = [enc_output[i] for i in bos_id_list[0] if i != -2]
            sentence_output = []
            #print(bos_ids, enc_last_hs_vectors.shape)
            for i in bos_ids[0].tolist():
                #print(i)
                if i != -2:
                    #print(i)
                    sentence_output.append(enc_last_hs_vectors[i].tolist())
            vector_list += sentence_output
        
        vector_list_pad = [0] * embed_dim
        vector_attn_pad = [0] * (max_len - len(vector_list))
        vector_attention = [1] * len(vector_list)

        vector_list += [vector_list_pad] * (max_len - len(vector_list))
        vector_attention += vector_attn_pad

        vector_list = torch.as_tensor([vector_list], device = encoder_output_list[0][0].device)
        #vector_attention = [1] * len(vector_list)
        #vector_attention = torch.as_tensor([vector_attention])
        #print("SENT VECT,  SENT ATTN", vector_list.shape, vector_attention.shape)
        return vector_list, vector_attention

    def _get_attention_masks_OR(self, 
        attention_mask_list ):

            #all_attn_outputs = torch.cat(attention_mask_list, 1)

            all_attn_outputs = torch.stack(attention_mask_list, 0)
            added_enc_attns = torch.Tensor.float(all_attn_outputs).mean(0).tolist()
            added_enc_attns = [[1 if each > 0 else 0 for each in each_list] for each_list in added_enc_attns]
            #added_enc_attns = torch.as_tensor([added_enc_attns])
            added_enc_attns = torch.as_tensor(added_enc_attns , device = attention_mask_list[0].device)
            return added_enc_attns

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
        bos_ids_col0 = None,
        bos_ids_col1 = None,
        bos_ids_col2 = None,
        bos_ids_col3 = None,
        bos_ids_col4 = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs_col0 = None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        encoder_outputs_HAT = None,
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
        #print(encoder_forward_strategy, encoder_combination_type)
        #print(encoder_outputs_col0)
        #print(attention_mask_col0)

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

        encoder_outputs = [encoder_outputs_col0, encoder_outputs_col1, encoder_outputs_col2, encoder_outputs_col3, encoder_outputs_col4]
        encoder_outputs_list = [each for each in encoder_outputs if each is not None]
        attention_masks = [attention_mask_col0, attention_mask_col1, attention_mask_col2, attention_mask_col3, attention_mask_col4]
        attn_mask_list = [each for each in attention_masks if each is not None]
        bos_id_list = [bos_ids_col0, bos_ids_col1, bos_ids_col2, bos_ids_col3, bos_ids_col4]

        if len(encoder_outputs_list) == 1:
            encoder_outputs = encoder_outputs_list[0]
            attn_mask = attn_mask_list[0] if attn_mask_list else None

        elif encoder_combination_type =='addition':
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

        elif encoder_combination_type == 'self_attention':
            encoder_outputs_concat = self._get_concat_encoder_outputs(encoder_outputs_list)
            attn_mask = self._get_attention_masks_OR(
                        [attn_mask for attn_mask in attn_mask_list if not (attn_mask is None)]

                    )
            attention_mask = _expand_mask(attn_mask, torch.float32, tgt_len=encoder_outputs_concat[0].shape[1])
            encoder_outputs = self._get_self_attn(encoder_outputs_concat, attention_mask, encoder_outputs_list)
            attn_mask = None
            


        if not encoder_outputs_HAT:
                sentence_representations, sentence_attention_mask = self._get_sentence_vectors(encoder_outputs_list, bos_id_list)
                sentence_attention_mask = torch.as_tensor([sentence_attention_mask], device = attention_mask_col0.device)
                encoder_outputs_HAT = self.hierarchical_attn_forward(sentence_representations, sentence_attention_mask)
                #print(sentence_representations, sentence_attention_mask)

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
            sentence_hidden_states = encoder_outputs_HAT[0],
            sentence_attention_mask = sentence_attention_mask,
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
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
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
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past  

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask_col0 = None,
        attention_mask_col1 = None,
        attention_mask_col2 = None,
        attention_mask_col3 = None,
        attention_mask_col4 = None,
        bos_ids_col0 = None,
        bos_ids_col1 = None,
        bos_ids_col2 = None,
        bos_ids_col3 = None,
        bos_ids_col4 = None,
        head_mask=None,
        use_cache=None,
        encoder_outputs_col0 =None,
        encoder_outputs_col1 = None,
        encoder_outputs_col2 = None,
        encoder_outputs_col3 = None,
        encoder_outputs_col4 = None,
        encoder_outputs_HAT = None,

        encoder_combination_type = 'HAT',
        decoder_attention_mask = None,
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
            "encoder_outputs_HAT": encoder_outputs_HAT,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask_col0": attention_mask_col0,
            "attention_mask_col1": attention_mask_col1,
            "attention_mask_col2": attention_mask_col2,
            "attention_mask_col3": attention_mask_col3,
            "attention_mask_col4": attention_mask_col4,
            "bos_ids_col0" : bos_ids_col0,
            "bos_ids_col1" : bos_ids_col1,
            "bos_ids_col2" : bos_ids_col2,
            "bos_ids_col3" : bos_ids_col3,
            "bos_ids_col4" : bos_ids_col4,
            "encoder_combination_type": encoder_combination_type,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            
        }        



    
