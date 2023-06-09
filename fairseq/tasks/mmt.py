# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from os.path import join, abspath, exists

from fairseq import options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    indexed_dataset,
    # VisionLanguageTripletDataset,
    PhraseLanguageTripletDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from . import FairseqTask, register_task

def load_orig_images(images_root, split):
    if split == 'test':
        split = 'test_2016_flickr'
    elif split == 'test1':
        split = 'test_2017_flickr'
    elif split == 'test2':
        split = 'test_2017_mscoco'
    parent_path = abspath(join(images_root, ".."))
    images_root = join(images_root, split)
    images_split_root = join(parent_path, 'image_splits')
    index = join(images_split_root, '{split}.txt'.format(split=split))
    if not exists(index):
        raise(RuntimeError("{0}.txt does not exist in {1}".format(split, images_split_root)))

    image_files = []
    with open(index, 'r') as f:
        for fname in f:
            fname = join(images_root, fname.strip())
            assert exists(fname), "{} does not exist.".format(fname)
            image_files.append(str(fname))
    return image_files

def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False,
        truncate_source=False,
        images_root=None, img_resize=None, img_color=None,
):
    list_images_paths = load_orig_images(images_root, split)

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    # vision_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            # vision_prefix = os.path.join(data_path, '{}.vision.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            # vision_prefix = os.path.join(data_path, '{}.vision.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        # vision_datasets.append(
        #     data_utils.load_indexed_dataset(vision_prefix + src, dictionary=None, dataset_impl=dataset_impl))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        # vision_dataset = vision_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return PhraseLanguageTripletDataset(
        src_dataset, src_dataset.sizes, src_dict,
        # vision_dataset,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        data_path=data_path,
        split=split,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        image_path=list_images_paths,
        image_resize=img_resize,
        image_color=img_color,
    )


@register_task('mmt')
class MultiModalTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', default=False, action='store_true',
                            help='boolean to truncate source to max-source-positions')
        parser.add_argument("--freeze_topk_update", default=0, type=int)
        # self-defined
        parser.add_argument('--image-path',metavar='STR',help='where save all_images')
        parser.add_argument('--image-resize', type=int,
                            help='resize of images')
        parser.add_argument('--image-color', type=float,
                            help='recolor of images')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
            images_root=self.args.image_path,
            img_resize=getattr(self.args, 'image_resize', None),
            img_color=getattr(self.args, 'image_color', None),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        list_images_paths = load_orig_images(self.args.image_path, split)
        return PhraseLanguageTripletDataset(src_tokens, src_lengths, self.source_dictionary, \
            image_path=list_images_paths,
            )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            # arch = getattr(args, 'model')
            # print("arch using at generator:", arch)
            # if 'doubly' in arch:
            #     from fairseq.doubly_generator import DoublyGenerator
            #     gen_cls = DoublyGenerator
            #     return gen_cls(
            #         self.target_dictionary,
            #         beam_size=getattr(args, 'beam', 5),
            #         max_len_a=getattr(args, 'max_len_a', 0),
            #         max_len_b=getattr(args, 'max_len_b', 200),
            #         min_len=getattr(args, 'min_len', 1),
            #         normalize_scores=(not getattr(args, 'unnormalized', False)),
            #         len_penalty=getattr(args, 'lenpen', 1),
            #         unk_penalty=getattr(args, 'unkpen', 0),
            #         sampling=getattr(args, 'sampling', False),
            #         sampling_topk=getattr(args, 'sampling_topk', -1),
            #         sampling_topp=getattr(args, 'sampling_topp', -1.0),
            #         temperature=getattr(args, 'temperature', 1.),
            #         diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            #         diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            #         match_source_len=getattr(args, 'match_source_len', False),
            #         no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            #     )
            from fairseq.sequence_generator import SequenceGenerator
            seq_gen_cls = SequenceGenerator
            print('using standard sequence generator')
            return seq_gen_cls(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
    

