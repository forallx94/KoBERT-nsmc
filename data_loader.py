import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NsmcProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # columns name 넘기고 진행 번
        for (i, line) in enumerate(lines[1:]):
            line = line.split('\t')
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = int(line[2])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            해당 모드의 데이터 로드
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))

        # 파일 처리 결과를 알기 쉽게 분리
        file_path = os.path.join(self.args.data_dir, file_to_read) # 경로 설정
        read_file_ = self._read_file(file_path) # text 파일 로드
        data_format_ = self._create_examples(read_file_, mode) # train 형식에 맞추어 데이터 변경
        return data_format_


processors = {
    "nsmc": NsmcProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    # 주요 token 설정
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokenize 진행
        tokens = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count: # 50 - 2
            tokens = tokens[:(max_seq_len - special_tokens_count)] # 앞의 48 토큰만 선택

        # Add [SEP] token
        # token_type_ids를 위한 위치 정보를 생성
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        # token_type_ids를 위한 위치 정보를 생성
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # token화된 단어를 id로 변경
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # 실 단어이면 attention_mask 1 아니면 0
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids) # 얼마나 비어있는지 
        input_ids = input_ids + ([pad_token_id] * padding_length) # 남은만큼 padding 추가
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length) # 남은 attention_mask 0
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    # NsmcProcessor load
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    # cached_nsmc_kobert_50_train
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    # ./data/cached_nsmc_kobert_50_train
    cached_features_file = os.path.join(args.data_dir, cached_file_name)

    # 파일이 존재할 경우 
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    # 파일이 없을 경우
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        # 데이터를 가져오는 과정

        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        #가져온 데이터를 feature로 변경
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)

        # 변경된 feature를 저장
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    # 변경한 데이터를 가져와서 torch.tensor로 데이터 설정
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    return dataset
