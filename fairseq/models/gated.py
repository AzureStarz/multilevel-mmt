# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import math
import numpy as np
import os

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import _load_weights

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    MultiheadAttention,
)
import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('gated')
class GatedModel(FairseqEncoderDecoderModel):

    def __init__(self, args, encoder, decoder, img_encoder):
        super().__init__(encoder, decoder, img_encoder)
        self.args = args
        self.epoch = 0

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # fmt: on
        # parser.add_argument('--visual_feature_file', default=None)
        # self-defined arguments
        # latent embedding (region embedding)
        parser.add_argument('--latent-embedding', type=str, metavar='STR',
                            help='latent embedding file')
        # phrase embedding (bert embedding)
        parser.add_argument('--phrase-embedding', type=str, metavar='STR',
                            help='phrase embedding file')
        # image embedding
        # parser.add_argument('--vision-embedding', type=str, metavar='STR',
        #                     help='phrase embedding file')
        parser.add_argument('--img-dim', type=int,help='size of image embedded dimension')
        parser.add_argument('--train-vision-model', action='store_true', default=False,
                            help='whether train vision model')
        parser.add_argument('--img-model',type=str,
                            help='the name of the image model in timm')
        parser.add_argument('--pretrain-weight',type=str,
                            help='the path of the pretrained image model weight')



    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        img_encoder = cls.build_img_encoder(args)
        return cls(args, encoder, decoder, img_encoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

    @classmethod
    def build_img_encoder(cls, args):
        return ImgEncoder(args)

    def forward(self, src_tokens, src_lengths, phrase_tokens, phrase_lengths, prev_output_tokens, **kwargs):
        self.encoder.epoch = self.epoch
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, phrase_tokens=phrase_tokens, phrase_lengths=phrase_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


class ImgEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = 0.5
        if "vit" in args.img_model:
            self.model = timm.create_model(args.img_model, pretrained=False, num_classes=0)
            _load_weights(self.model, args.pretrain_weight)
        else:
            self.model = timm.create_model(args.img_model, pretrained=True, num_classes=0)
        self.img_dim = args.img_dim
        self.num_features = self.model.num_features
        self.fc = nn.Linear(self.num_features, self.img_dim)
        # if args.img_model_type == 'cnn':
        #     self.global_pool, self.fc = create_classifier(self.num_features, self.img_dim, pool_type='avg')
        # elif args.img_model_type == 'transformer':
        #     self.embed_dim = self.num_features
        #     self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #     self.fc_norm = self.norm_layer(self.embed_dim)
        #     self.head = nn.Linear(self.embed_dim, self.img_dim)
        # else:
        #     raise NotImplementedError()
        if args.train_vision_model:
            self.model.train()
        else: 
            for param in self.model.parameters():
                param.requires_grad=False

    def forward(self, x):
        '''
        x: image tensor: [batch, 3, 224, 224]
        '''
        img_features = self.model(x)
        return self.fc(img_features)
        # if self.args.img_model_type == 'cnn':
            # global features
        #     img_features = self.global_pool(img_features)
        #     if self.dropout:
        #         img_features = F.dropout(img_features, p=float(self.dropout), training=self.training)
        #     return self.fc(img_features)
        # elif self.args.img_model_type == 'transformer':
            # img_features = img_features[:, self.model.num_prefix_tokens:].mean(dim=1)
        #     img_features = img_features[:, 0]
        #     img_features = self.fc_norm(img_features)
        #     return self.head(img_features)
        # else:
        #     raise NotImplementedError()

EncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_embedding',  # B x T x C
    'encoder_states',  # List[T x B x C]
])


class TransformerEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.epoch = 0

        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, self.embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        # vision related
        # embeding_weights = np.load(args.vision_embedding)
        # img_vocab, self.img_dim = embeding_weights.shape
        # vision start from 1
        # embeddings_matrix = np.zeros((img_vocab + 1, self.img_dim))
        # embeddings_matrix[1:] = embeding_weights
        # shape: [img_vocab, img_dim]
        # self.visual_features = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings_matrix), freeze=True)  # update embedding
        self.img_dim = getattr(args, 'img_dim', 2048)
        self.dense = nn.Linear(self.img_dim, self.embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.lambda_dense = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gate_dense = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.args = args

        self.out = open(args.save_dir + '/gate.txt', 'w')

        # multilevel attn
        self.sentence_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            kdim=self.embed_dim,
            vdim=self.embed_dim,
            dropout=args.attention_dropout,
            bias=True,
            encoder_decoder_attention=True
        )
        self.phrase_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            kdim=self.embed_dim,
            vdim=self.embed_dim,
            dropout=args.attention_dropout,
            bias=True,
            encoder_decoder_attention=True
        )
        # phrase and region feature matrix
        self.latent_embed_mat = torch.tensor(np.load(args.latent_embedding), dtype=torch.float32).cuda()
        self.phrase_embed_mat = torch.tensor(np.load(args.phrase_embedding), dtype=torch.float32).cuda()
        self.fc_latent = Linear(self.latent_embed_mat.shape[-1], self.embed_dim)
        self.fc_phrase = Linear(self.phrase_embed_mat.shape[-1], self.embed_dim)
        

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def phrase_mask_matrix(self, phrase_lengths):
        batch_size = phrase_lengths.shape[0]
        max_length = torch.max(phrase_lengths)
        mask_matrix = torch.zeros((batch_size, max_length), dtype=torch.bool).cuda()
        for sid in range(batch_size):
            mask_matrix[sid, phrase_lengths[sid]:] = True
        return mask_matrix

    def phrase_embedding(
        self, phrase_tokens, phrase_lengths
    ):
        mask_matrix = self.phrase_mask_matrix(phrase_lengths)
        # phrase_tokens shape [batch_size, max_length]
        unfold_phrase_tokens = phrase_tokens.reshape(-1)
        unfold_phrase_mask = mask_matrix.reshape(-1)
        assert len(unfold_phrase_tokens) == len(unfold_phrase_mask)
        # acquire phrase_embedding
        unfold_phrase_embedding = torch.index_select(self.phrase_embed_mat, 0, unfold_phrase_tokens)
        unfold_phrase_embedding = self.fc_phrase(unfold_phrase_embedding)
        # shape [batch_size, max_length, embed_dim]
        phrase_embedding = unfold_phrase_embedding.reshape(phrase_tokens.shape[0], phrase_tokens.shape[1], self.embed_dim)
        # acquire region_embedding
        unfold_region_embedding = torch.index_select(self.latent_embed_mat, 0, unfold_phrase_tokens)
        unfold_region_embedding = self.fc_latent(unfold_region_embedding)
        # shape [batch_size, max_length, embed_dim]
        region_embedding = unfold_region_embedding.reshape(phrase_tokens.shape[0], phrase_tokens.shape[1], self.embed_dim)
        if self.layernorm_embedding:
            phrase_embedding = self.layernorm_embedding(phrase_embedding)
        phrase_embedding = F.dropout(phrase_embedding, p=self.dropout, training=self.training)
        if self.layernorm_embedding:
            region_embedding = self.layernorm_embedding(region_embedding)
        region_embedding = F.dropout(region_embedding, p=self.dropout, training=self.training)

        if torch.min(phrase_lengths) == 0:
            for sid in range(mask_matrix.shape[0]):
                if phrase_lengths[sid] == 0:
                    mask_matrix[sid, :] = False
        return phrase_embedding, region_embedding, mask_matrix

    def forward(self, src_tokens, src_lengths, phrase_tokens, phrase_lengths, img_features=None, cls_input=None, return_all_hiddens=False, **unused):

        batch_size = src_tokens.shape[0]
        max_phrase_length = torch.max(phrase_lengths)
        max_text_length = torch.max(src_lengths)

        if self.layer_wise_attention:
            return_all_hiddens = True

        if max_phrase_length > 0:
            # shape [batch_size, max_length, embed_dim]
            p, r, phrase_mask_matrix = self.phrase_embedding(phrase_tokens, phrase_lengths)

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        # image embedding
        # batch_visual_ids = visions.type(torch.LongTensor).to(src_tokens.device)
        # batch_size = batch_visual_ids.size()[0]
        # v_embedding = self.visual_features(batch_visual_ids)  # B*img_dim
        # print("v embedding shape before view:", v_embedding.shape)
        v_embedding = img_features.view(batch_size, 1, self.img_dim)  # B, 1, img_dim
        # print("v embedding shape:", v_embedding.shape)
        text_repr = x.transpose(0, 1)  # T x B x C -> B x T x C
        # v_repr = self.dense(v_embedding)  # B, 1, C

        # sentence-level attention
        
        sentence_repr = x.transpose(0, 1) # T x B x C -> B x T x C
        image_repr = self.dense(v_embedding) # B, 1, C
        b, t, c = sentence_repr.shape
        image_repr = image_repr.expand(b, t, c)
        assert image_repr.shape[1] == sentence_repr.shape[1]
        sentence_image_aware, attn = self.sentence_attn(
            query=sentence_repr,
            key=image_repr,
            value=image_repr,
        )
        
        
        # phrase-level attention
        
        # B * max_length * embedded_dim
        phrase_repr = p.transpose(0, 1)
        region_repr = r.transpose(0, 1)
        # print(phrase_mask_matrix)
        phrase_region_aware, attn = self.phrase_attn(
                query=phrase_repr,
                key=region_repr,
                value=region_repr,
                key_padding_mask=phrase_mask_matrix
        )
        
        
        # average phrase-level attention
        phrase_region_aware = phrase_region_aware.transpose(0, 1)
        multiply_mask = phrase_mask_matrix.unsqueeze(-1)
        multiply_mask = multiply_mask.expand(phrase_region_aware.shape[0],phrase_region_aware.shape[1],phrase_region_aware.shape[2])
        phrase_region_aware = torch.mul(phrase_region_aware, ~multiply_mask)
        phrase_region_aware[phrase_region_aware == 0] = float('nan')
        # [B, 1, embedded_dim]
        np_phrase_region_aware = phrase_region_aware.detach().cpu().numpy()
        np_phrase_region_aware = np.nanmean(np_phrase_region_aware, axis=1, keepdims=True)
        phrase_region_aware = torch.tensor(np_phrase_region_aware, dtype=torch.float32).cuda()
        # [B, T, embedded_dim]
        phrase_region_aware = phrase_region_aware.expand(b, t, c)
        

        # multi-level aggregation
        
        merge = torch.cat([sentence_image_aware, phrase_region_aware], dim=-1)
        lamb = self.sigmoid(self.lambda_dense(merge))
        multilevel_repr = sentence_image_aware + lamb * phrase_region_aware
        
        # gated fusion
        
        merge = torch.cat([text_repr, multilevel_repr], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        output = text_repr + gate * multilevel_repr
        
        
        # origin implement
        """
        b, t, c = text_repr.shape
        v_repr = v_repr.expand(b, t, c)
        assert v_repr.shape[1] == text_repr.shape[1]
        merge = torch.cat([text_repr, v_repr], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        # mask = src_tokens.ne(self.padding_idx).unsqueeze(-1).expand(b, t, c)
        # print(gate[mask].flatten().tolist(), file=self.out)
        # output = (1 - gate) * text_repr + gate * output  # for video 
        output = text_repr + gate * v_repr  # for image, standard one
        """

        x = output.transpose(0, 1)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            **unused,
    ):

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('gated', 'gated')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)


@register_model_architecture('gated', 'gated_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('gated', 'gated_tiny')
def transformer_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 256)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    base_architecture(args)


@register_model_architecture('gated', 'gated_vatex')
def uvr_video_vatex(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)

