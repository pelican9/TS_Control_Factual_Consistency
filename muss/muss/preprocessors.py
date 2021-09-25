# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from functools import wraps, lru_cache
import hashlib
from pathlib import Path
import dill as pickle
import shutil

import nevergrad as ng
import numpy as np
import sentencepiece as spm
# from fairseq.data.encoders.gpt2_bpe_utils import get_encoder

from muss.feature_extraction import (
    get_lexical_complexity_score,
    get_levenshtein_similarity,
    get_dependency_tree_depth,
    get_replace_only_levenshtein_similarity,
)
from muss.resources.paths import VARIOUS_DIR, RESOURCES_DIR
from muss.utils.resources import download
from muss.text import remove_multiple_whitespaces, extract_special_tokens
from muss.utils.helpers import (
    write_lines_in_parallel,
    yield_lines_in_parallel,
    add_dicts,
    get_default_args,
    get_temp_filepath,
    failsafe_division,
    count_lines,
    log_action,
    get_files_hash,
)

############################### from fairseq.data.encoders.gpt2_bpe_utils import get_encoder:


import json
from functools import lru_cache


from fairseq.data.encoders.gpt2_bpe_utils import bytes_to_unicode
# @lru_cache()
# def bytes_to_unicode():
#     """
#     Returns list of utf-8 byte and a corresponding list of unicode strings.
#     The reversible bpe codes work on unicode strings.
#     This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
#     When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
#     This is a signficant percentage of your normal, say, 32K bpe vocab.
#     To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
#     And avoids mapping to whitespace/control characters the bpe code barfs on.
#     """
#     bs = (
#         list(range(ord("!"), ord("~") + 1))
#         + list(range(ord("Â¡"), ord("Â¬") + 1))
#         + list(range(ord("Â®"), ord("Ã¿") + 1))
#     )
#     cs = bs[:]
#     n = 0
#     for b in range(2 ** 8):
#         if b not in bs:
#             bs.append(b)
#             cs.append(2 ** 8 + n)
#             n += 1
#     cs = [chr(n) for n in cs]
#     return dict(zip(bs, cs))
    

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        try:
            import regex as re

            self.re = re
        except ImportError:
            raise ImportError("Please install regex with: pip install regex")

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(
            r"""<NEXT_DIFFICULT_WORD>|<NEXT_NE>|<SENT_START>|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?(?!<)(?!_)(?!>)[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        special_token_list = ['<NEXT_DIFFICULT_WORD>','<NEXT_NE>','<SENT_START>']
        bpe_tokens = []

        # print('self.re.findall(self.pat, text)',self.re.findall(self.pat, text))
        for token in self.re.findall(self.pat, text):
            if token in special_token_list:
                # print('token',token)
                # print('self.re.findall(self.pat, text)',self.re.findall(self.pat, text))
                bpe_tokens.extend([self.encoder[token]])
            else:
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
                )

        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder.get(token, token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text
        
def get_encoder(encoder_json_path, vocab_bpe_path):
    
    print('encoder_json_path',encoder_json_path)
    print('vocab_bpe_path',vocab_bpe_path)
    
    
    with open(encoder_json_path, "r") as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )



###########################################################################
PREPROCESSORS_REGISTRY = {}


def get_preprocessor_by_name(preprocessor_name):
    return PREPROCESSORS_REGISTRY[preprocessor_name]


def get_preprocessors(preprocessors_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessors_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors


def remove_special_tokens(sentence):
    return extract_special_tokens(sentence)[1]


def store_args(constructor):
    @wraps(constructor)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, 'args') or not hasattr(self, 'kwargs'):
            # TODO: Default args are not overwritten if provided as args
            self.args = args
            self.kwargs = add_dicts(get_default_args(constructor), kwargs)
        return constructor(self, *args, **kwargs)

    return wrapped


def dump_preprocessors(preprocessors, dir_path):
    with open(Path(dir_path) / 'preprocessors.pickle', 'wb') as f:
        pickle.dump(preprocessors, f)


def load_preprocessors(dir_path):
    path = Path(dir_path) / 'preprocessors.pickle'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


class AbstractPreprocessor(ABC):
    def __init_subclass__(cls, **kwargs):
        '''Register all children in registry'''
        super().__init_subclass__(**kwargs)
        PREPROCESSORS_REGISTRY[cls.__name__] = cls

    def __repr__(self):
        args = getattr(self, 'args', ())
        kwargs = getattr(self, 'kwargs', {})
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [f'{k}={repr(v)}' for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])]
        args_kwargs_str = ', '.join(args_repr + kwargs_repr)
        return f'{self.__class__.__name__}({args_kwargs_str})'

    def get_hash_string(self):
        return self.__class__.__name__

    def get_hash(self):
        return hashlib.md5(self.get_hash_string().encode()).hexdigest()

    @staticmethod
    def get_nevergrad_variables():
        return {}

    @property
    def prefix(self):
        return self.__class__.__name__.replace('Preprocessor', '')

    def fit(self, complex_filepath, simple_filepath):
        pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def decode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        if complex_sentence is not None:
            complex_sentence = self.encode_sentence(complex_sentence)
        if simple_sentence is not None:
            simple_sentence = self.encode_sentence(simple_sentence)
        return complex_sentence, simple_sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w') as f:
            for input_line, encoder_line in yield_lines_in_parallel([input_filepath, encoder_filepath], strict=False):
                f.write(self.encode_sentence(input_line, encoder_line) + '\n')

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w') as f:
            for encoder_sentence, input_sentence in yield_lines_in_parallel(
                [encoder_filepath, input_filepath], strict=False
            ):
                decoded_sentence = self.decode_sentence(input_sentence, encoder_sentence=encoder_sentence)
                f.write(decoded_sentence + '\n')

    def encode_file_pair(self, complex_filepath, simple_filepath, output_complex_filepath, output_simple_filepath):
        '''Jointly encode a complex file and a simple file (can be aligned or not)'''
        with write_lines_in_parallel([output_complex_filepath, output_simple_filepath], strict=False) as output_files:
            for complex_line, simple_line in yield_lines_in_parallel([complex_filepath, simple_filepath], strict=False):
                output_files.write(self.encode_sentence_pair(complex_line, simple_line))


class ComposedPreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(self, preprocessors, sort=False):
        if preprocessors is None:
            preprocessors = []
        if sort:
            # Make sure preprocessors are always in the same order
            preprocessors = sorted(preprocessors, key=lambda preprocessor: preprocessor.__class__.__name__)
        self.preprocessors = preprocessors

    def get_hash_string(self):
        preprocessors_hash_strings = [preprocessor.get_hash_string() for preprocessor in self.preprocessors]
        return f'ComposedPreprocessor(preprocessors={preprocessors_hash_strings})'

    def get_suffix(self):
        return '_'.join([p.prefix.lower() for p in self.preprocessors])

    def fit(self, complex_filepath, simple_filepath):
        for preprocessor in self.preprocessors:
            pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.encode_sentence(sentence, encoder_sentence)
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.decode_sentence(sentence, encoder_sentence)
        return sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.encode_file(input_filepath, intermediary_output_filepath, encoder_filepath)
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.decode_file(input_filepath, intermediary_output_filepath, encoder_filepath)
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def encode_file_pair(self, complex_filepath, simple_filepath, output_complex_filepath, output_simple_filepath):
        for preprocessor in self.preprocessors:
            intermediary_output_complex_filepath = get_temp_filepath()
            intermediary_output_simple_filepath = get_temp_filepath()
            preprocessor.encode_file_pair(
                complex_filepath,
                simple_filepath,
                intermediary_output_complex_filepath,
                intermediary_output_simple_filepath,
            )
            complex_filepath = intermediary_output_complex_filepath
            simple_filepath = intermediary_output_simple_filepath
        shutil.copyfile(complex_filepath, output_complex_filepath)
        shutil.copyfile(simple_filepath, output_simple_filepath)

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        for preprocessor in self.preprocessors:
            complex_sentence, simple_sentence = preprocessor.encode_sentence_pair(complex_sentence, simple_sentence)
        return complex_sentence, simple_sentence


class FeaturePreprocessor(AbstractPreprocessor):
    '''Prepend a computed feature at the beginning of the sentence'''

    @store_args
    def __init__(
        self,
        feature_name,
        get_feature_value,
        get_target_feature_value,
        bucket_size=0.05,
        noise_std=0,
        prepend_to_target=False,
        use_short_name=False,
    ):
        self.get_feature_value = get_feature_value
        self.get_target_feature_value = get_target_feature_value
        self.bucket_size = bucket_size
        self.noise_std = noise_std
        self.feature_name = feature_name.upper()
        self.use_short_name = use_short_name
        if use_short_name:
            # There might be collisions
            self.feature_name = self.feature_name.lower()[:4]
        self.prepend_to_target = prepend_to_target

    def get_hash_string(self):
        return f'{self.__class__.__name__}(feature_name={repr(self.feature_name)}, bucket_size={self.bucket_size}, noise_std={self.noise_std}, prepend_to_target={self.prepend_to_target}, use_short_name={self.use_short_name})'  # noqa: E501

    def bucketize(self, value):
        '''Round value to bucket_size to reduce the number of different values'''
        return round(round(value / self.bucket_size) * self.bucket_size, 10)

    def add_noise(self, value):
        return value + np.random.normal(0, self.noise_std)

    def get_feature_token(self, feature_value):
        return f'<{self.feature_name}_{feature_value}>'

    def encode_sentence(self, sentence, encoder_sentence=None):
        if not self.prepend_to_target:
            desired_feature = self.bucketize(self.get_target_feature_value(remove_special_tokens(sentence)))
            sentence = f'{self.get_feature_token(desired_feature)} {sentence}'
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        if self.prepend_to_target:
            _, sentence = extract_special_tokens(sentence)
        return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        feature = self.bucketize(
            self.add_noise(
                self.get_feature_value(remove_special_tokens(complex_sentence), remove_special_tokens(simple_sentence))
            )
        )
        if self.prepend_to_target:
            simple_sentence = f'{self.get_feature_token(feature)} {simple_sentence}'
        else:
            complex_sentence = f'{self.get_feature_token(feature)} {complex_sentence}'
        return complex_sentence, simple_sentence


class LevenshteinPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    @staticmethod
    def get_nevergrad_variables():
        return {'target_ratio': ng.p.Scalar(init=0.8, lower=0.2, upper=1)}

    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_levenshtein_similarity(complex_sentence, simple_sentence)

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class ReplaceOnlyLevenshteinPreprocessor(LevenshteinPreprocessor):
    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_replace_only_levenshtein_similarity(complex_sentence, simple_sentence)


class RatioPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, feature_extractor, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    @staticmethod
    def get_nevergrad_variables():
        return {'target_ratio': ng.p.Scalar(init=0.8, lower=0.2, upper=1.5)}

    def get_feature_value(self, complex_sentence, simple_sentence):
        return min(
            failsafe_division(self.feature_extractor(simple_sentence), self.feature_extractor(complex_sentence)), 2
        )

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class LengthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(len, *args, **kwargs)


class WordRankRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(lambda sentence: get_lexical_complexity_score(sentence, language=language), *args, **kwargs)


class DependencyTreeDepthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(lambda sentence: get_dependency_tree_depth(sentence, language=language), *args, **kwargs)


def train_sentencepiece(input_filepaths, vocab_size, sentencepiece_model_path, num_threads=64, max_lines=10 ** 7):
    with log_action('Training sentencepiece'):
        sentencepiece_model_path.parent.mkdir(parents=True, exist_ok=True)
        sentencepiece_model_prefix = sentencepiece_model_path.parent / sentencepiece_model_path.stem
        args_str = f'''
        --bos_id=-1 --eos_id=-1
        --input={",".join([str(path) for path in input_filepaths])} --model_prefix={sentencepiece_model_prefix}
        --vocab_size={vocab_size} --num_threads={num_threads} --character_coverage=0.9995
        '''
        if sum([count_lines(filepath) for filepath in input_filepaths]) > max_lines:
            args_str += f' --input_sentence_size={max_lines} --shuffle_input_sentence=true'
        args_str = remove_multiple_whitespaces(args_str.replace('\n', ' ')).strip(' ')
        spm.SentencePieceTrainer.Train(args_str)
        return sentencepiece_model_path


def load_sentencepiece_model(sentencepiece_model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sentencepiece_model_path))
    return sp


def write_sentencepiece_vocab_as_fairseq_dict(sentencepiece_model, fairseq_dict_path=None):
    if fairseq_dict_path is None:
        fairseq_dict_path = get_temp_filepath()
    with open(fairseq_dict_path, 'w') as f:
        for i in range(len(sentencepiece_model)):
            piece = sentencepiece_model.id_to_piece(i)
            if piece.startswith('<') and piece.endswith('>'):
                continue
            f.write(f'{piece} 999\n')  # Use 999 as dummy count
        return fairseq_dict_path


class SentencePiecePreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(
        self,
        vocab_size=None,
        input_filepaths=None,
        num_threads=64,
        max_lines=10 ** 7,
        sentencepiece_model_path=None,
        tokenize_special_tokens=False,
    ):
        self.tokenize_special_tokens = tokenize_special_tokens
        if sentencepiece_model_path is None:
            sentencepiece_model_path = (
                VARIOUS_DIR
                / f'sentencepiece_model/spm_files-{get_files_hash(input_filepaths)}_lines-{max_lines}_vocab-{vocab_size}.model'
            )  # noqa: E501
            if not sentencepiece_model_path.exists():
                train_sentencepiece(input_filepaths, vocab_size, sentencepiece_model_path)
        self.sentencepiece_model_path = sentencepiece_model_path

    @property
    @lru_cache(maxsize=1)
    def sp(self):
        """
        We need to use a property because SentencenPieceProcessor cannot pickled
        > pickle.dumps(spm.SentencePieceProcessor())
        ----> TypeError: can't pickle SwigPyObject objects
        """
        return load_sentencepiece_model(self.sentencepiece_model_path)

    def get_hash_string(self):
        return f'{self.__class__.__name__}(sentencepiece_model_path={self.sentencepiece_model_path}, tokenize_special_tokens={self.tokenize_special_tokens})'

    def encode_sentence(self, sentence, encoder_sentence=None):
        # TODO: Do we really need to extract the tokens
        special_tokens = ''
        if not self.tokenize_special_tokens:
            special_tokens, sentence = extract_special_tokens(sentence)
        # Instead of encoding directly to pieces with self.sp.EncodeAsPieces, we first encode as ids to enforce <unk> tokens when a token is not in the vocab
        spm_ids = self.sp.EncodeAsIds(sentence)
        encoded_sentence = ' '.join(self.sp.id_to_piece(spm_id) for spm_id in spm_ids)
        if special_tokens != '':
            encoded_sentence = f'{special_tokens} {encoded_sentence}'
        return encoded_sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        return self.sp.DecodePieces(sentence.split(' '))

    def create_fairseq_dict(self):
        return write_sentencepiece_vocab_as_fairseq_dict(self.sp)

    def copy_sentencepiece_files_to_dir(self, dirpath):
        shutil.copy(self.sentencepiece_model_path, dirpath)  # Copy with original name
        shutil.copyfile(
            self.sentencepiece_model_path, dirpath / 'sentencepiece.bpe.model'
        )  # Copy with standardized name
        shutil.copyfile(self.create_fairseq_dict(), dirpath / 'dict.txt')


class GPT2BPEPreprocessor(AbstractPreprocessor):
    def __init__(self):
        self.bpe_dir = RESOURCES_DIR / 'bart_bpe'
        self.bpe_dir.mkdir(exist_ok=True, parents=True)
        self.encoder_json_path = self.bpe_dir / 'encoder.json'
        self.vocab_bpe_path = self.bpe_dir / 'vocab.bpe'
        self.dict_path = self.bpe_dir / 'dict.txt'
        download(
            'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json', self.encoder_json_path, overwrite=False
        )
        download('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe', self.vocab_bpe_path, overwrite=False)
        download('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt', self.dict_path, overwrite=False)

    @property
    @lru_cache(maxsize=1)
    def bpe_encoder(self):
        """
        We need to use a property because GPT2BPEPreprocessor() is cannot pickled
        > pickle.dumps(GPT2BPEPreprocessor())
        ----> TypeError: can't pickle module objects
        """
        return get_encoder(self.encoder_json_path, self.vocab_bpe_path)

    def encode_sentence(self, sentence, *args, **kwargs):
        return ' '.join([str(idx) for idx in self.bpe_encoder.encode(sentence)])

    def decode_sentence(self, sentence, *args, **kwargs):
        ####################
        # list_of_id = [int(idx) for idx in sentence.split(' ')]
        list_of_id = []
        for idx in sentence.split(' '):
          try:
            int(idx)
            list_of_id.append(int(idx))
          except ValueError:
            pass  
            # print('decode error','idx,sentence:\n',idx,sentence)      # or whatever
        
        # decode error idx,sentence:
          
        # error is two white spaces
  
        ####################
        return self.bpe_encoder.decode(list_of_id)
