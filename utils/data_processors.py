import os
import json
import random
import re
import warnings
import dataclasses
from dataclasses import dataclass
from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Optional, Union

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers.file_utils import is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers import DataProcessor, InputExample, AlternatingCodebooksLogitsProcessor
from utils.config import args
from ALconfig import ALargs

if is_tf_available():
    import tensorflow as tf

logger = logging.get_logger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)
DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py"
)


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
            examples: tf.data.Dataset,
            tokenizer: PreTrainedTokenizer,
            task=str,
            max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
        label_type = tf.float32 if task == "sts-b" else tf.int64

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = tokenizer.model_input_names

        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, label_type),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )


def convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_map: Optional[Dict] = None,
        output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.

    """
    warnings.warn(DEPRECATION_WARNING.format("function"), FutureWarning)
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_map=label_map, output_mode=output_mode
    )


def _convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_map: Optional[Dict] = None,
        output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = processors[task]()
        if label_map is None:
            label_map = processor.get_label_map()
            logger.info(f"Using label list {label_map} for task {task}")
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_ids = []
    for k, v in label_map.items():
        assert len(tokenizer.tokenize(' ' + v)) == 1
        label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_ids.append(label_id[0])

    label_map: Dict[str, int] = {label_value: label_id for label_value, label_id in zip(label_map.values(), label_ids)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels: List[int] = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        # padding="max_length",
        padding=True,
        truncation=True,
        return_token_type_ids=True,
    )

    features = []
    for i in tqdm(range(len(examples)), desc="Converting examples to features", position=0, leave=True,
                  total=len(examples)):
        inputs: Dict[str, List[int]] = {k: batch_encoding[k][i] for k in batch_encoding}
        input_ids: List[int] = inputs['input_ids']

        try:
            # Mask token is a special token present in the input_ids used to mask the token to be predicted
            mask_pos = input_ids.index(tokenizer.mask_token_id)
        except:
            seq_len = len(input_ids)
            mask_pos = seq_len - 2
            if seq_len >= max_length:
                input_ids[-2] = tokenizer.mask_token_id
                inputs['input_ids'] = input_ids
            else:
                input_ids = input_ids[0:-1] + [tokenizer.mask_token_id] + [input_ids[-1]]
                attention_mask = inputs['attention_mask'] + [1]
                token_type_ids = inputs['token_type_ids'] + [tokenizer.pad_token_type_id]
                inputs = {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                }

        feature = InputFeatures(**inputs, label=labels[i], mask_pos=mask_pos)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example.guid}")
        logger.info(f"features: {features[i]}")

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "No", "1": "Yes"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            text_a = '{} ? <mask> , {}'.format(text_a, text_b)
            label = None if set_type == "test" else label_map[line[0]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, args):
        """See base class."""
        data_dir = args.data_dir
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_label_map(self):
        label_map = {"contradiction": 'No', "entailment": 'Yes', "neutral": 'Maybe'}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[8]
            text_b = line[9]
            text_a = '{} ? <mask> , {}'.format(text_a, text_b)
            label = None if set_type.startswith("test") else label_map[line[-1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        ## e:list: [['sentence', 'label'],['hide new secretions from the parental units ', '0']...]
        # e = self._read_tsv(os.path.join(data_dir, "train.tsv"))

        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "terrible", "1": "great"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_a = line[text_index]
            text_a = '{} . It was <mask> .'.format(text_a)

            text_a = text_a.replace('\\n', ' ').replace('\\', ' ').strip()

            label = None if set_type == "test" else label_map[line[1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class ImdbProcessor(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str, args=None):
        res_line = []
        res_line.append(['sentence', 'label'])

        if set_type == "test":
            test_path = os.path.join(project_root, "data/imdb/test.json")
            if args.ood:
                if args.ood_name == 'sst2':
                    test_path = os.path.join(project_root,'data/sst2val.json')
                elif args.ood_name == 'imdb_contrast':
                    test_path = os.path.join(project_root,'data/IMDB-contrast.json')
                elif args.ood_name == 'imdb_counter':
                    test_path = os.path.join(project_root,'data/IMDB-counter.json')
                else:
                    raise FileNotFoundError(f"The specified path '{test_path}' does not exist.")

            print('imdb test path!!!:', test_path)
            with open(test_path, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "train":

            with open(os.path.join(project_root, "data/imdb/train.json")) as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "dev":

            with open(os.path.join(project_root, "data/imdb/dev.json")) as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, args):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test', args=args), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "terrible", "1": "great"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_a = line[text_index]


            # raw_max_len = 256  - 18
            # text_a = text_a[:raw_max_len]
            text_a = '{} . It was <mask> .'.format(text_a)
            text_a = text_a.replace('\\n', ' ').replace('\\', ' ').strip()
            label = label_map[str(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": 'No', "1": 'Yes'}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                text_a = '{} <mask> , {}'.format(text_a, text_b)
                label = None if test_mode else label_map[line[5]]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def get_label_map(self):
        label_map = {'entailment': 'Yes', 'not_entailment': 'No'}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            text_a = '{} ? <mask> , {}'.format(text_a, text_b)
            label = None if set_type == "test" else label_map[line[-1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def get_label_map(self):
        label_map = {'entailment': 'Yes', 'not_entailment': 'No'}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            text_a = '{} ? <mask> , {}'.format(text_a, text_b)
            label = None if set_type == "test" else label_map[line[-1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MpqaProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "terrible", "1": "great"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            text_a = '{} . It was <mask> .'.format(text_a)
            label = None if set_type == "test" else label_map[line[0]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class MrProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "terrible", "1": "great"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            text_a = '{} . It was <mask> .'.format(text_a)
            label = None if set_type == "test" else label_map[line[0]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class SubjProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_map(self):
        label_map = {"0": "subjective", "1": "objective"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            text_a = '{} . It was <mask> .'.format(text_a)
            label = None if set_type == "test" else label_map[line[0]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


tasks_num_labels = {
    "mnli": 3,
    "mrpc": 2,
    "sst2": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "mpqa": 2,
    "mr": 2,
    "subj": 2,

    ### AL dataset
    "imdb": 2,
    "agnews": 4,
    "yelp": 5,
    "yahoo": 10,
    "dbpedia": 14,
    "trec": 6,

}


class AgnewsProcessor(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str):
        res_line = []
        res_line.append(['sentence', 'label'])
        # train_pool_labels = []
        with open(os.path.join(project_root, f"data/agnews/{set_type}.json")) as f:
            lines = f.readlines()
        for i in lines:
            i = json.loads(i)
            res_line.append([i["text"], str(i["_id"])])
            # train_pool_labels.append(i["_id"])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test'), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_label_map(self):
        label_map = {"0": "world", "1": "sports", "2": "business", "3": "technology"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_a = line[text_index]

            ### 根据最大长度进行截断：
            # raw_max_len = 256  - 18
            # text_a = text_a[:raw_max_len]
            text_a = '{} . It was <mask> .'.format(text_a)
            text_a = text_a.replace('\\n', ' ').replace('\\', ' ').strip()
            label = label_map[str(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class YelpProcessor(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str):
        res_line = []
        res_line.append(['sentence', 'label'])

        if set_type == "test":
            with open(os.path.join(project_root, "data/yelp/test.json")) as f:
                lines = f.readlines()

            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "train":
            with open(os.path.join(project_root, "data/yelp/train.json")) as f:
                lines = f.readlines()

            # train_pool_labels = []
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])
            #     train_pool_labels.append(i["_id"])
            # ALargs.class_labels  = train_pool_labels

        if set_type == "dev":

            with open(os.path.join(project_root, "data/yelp/dev.json")) as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test'), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def get_label_map(self):
        label_map = {"0": "terrible", "1": "bad", "2": "okay", "3": "good", "4": "great"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_a = line[text_index]


            text_a = '{} . It was <mask> .'.format(text_a)

            text_a = text_a.replace('\\n', ' ').replace('\\', ' ').strip()
            label = label_map[str(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class YahooProcessor(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str):
        res_line = [['sentence', 'label']]

        if set_type == "test":
            with open(os.path.join(project_root, "data/yahoo/test.json")) as f:
                lines = f.readlines()

            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "train":
            with open(os.path.join(project_root, "data/yahoo/train_10wan.json")) as f:
                lines = f.readlines()
            # train_pool_labels = []
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])
            #     train_pool_labels.append(i["_id"])
            # ALargs.class_labels = train_pool_labels

        if set_type == "dev":

            with open(os.path.join(project_root, "data/yahoo/dev_2wan.json")) as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test'), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def get_label_map(self):
        label_map = {
            "0": "society",
            "1": "science",
            "2": "health",
            "3": "education",
            "4": "computer",
            "5": "sports",
            "6": "business",
            "7": "entertainment",
            "8": "relationship",
            "9": "politics"
        }

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_ = line[text_index]
            text = text_.split("  ")  # 以两个空格为分隔符，将 `text` 分割为多个部分
            title, body = text[0], " ".join(text[1:])  # 第一个部分作为标题，其余部分合并为正文

            text_title = title.replace('\\n', ' ').replace('\\', ' ').strip()


            text_content = body.replace('\\n', ' ').replace('\\', ' ').strip()


            text_a = f'[ Category : <mask> ] {text_title}{text_content}'

            label = label_map[str(line[1])]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class DBPediaProcessor(DataProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str):
        res_line = [['sentence', 'label']]

        if set_type == "test":
            with open(os.path.join(project_root, "data/dbpedia/test.json")) as f:
                lines = f.readlines()

            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "train":
            with open(os.path.join(project_root, "data/dbpedia/train_14wan.json")) as f:
                lines = f.readlines()
            # train_pool_labels =[]
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])
            #     train_pool_labels.append(i["_id"])
            # ALargs.class_labels = train_pool_labels

        if set_type == "dev":
            # 2万个样本
            with open(os.path.join(project_root, "data/dbpedia/dev_2wan.json")) as f:
                lines = f.readlines()
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test'), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

    def get_label_map(self):
        # label_map = {"0": "world", "1": "sports", "2": "business", "3": "technology"}
        label_map = {
            "0": "company",
            "1": "school",
            "2": "artist",
            "3": "athlete",
            "4": "politics",
            "5": "transportation",
            "6": "building",
            "7": "mountain",
            "8": "village",
            "9": "animal",
            "10": "plant",
            "11": "album",
            "12": "film",
            "13": "book"
        }

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_ = line[text_index]
            text = text_.split("  ")
            title, body = text[0], " ".join(text[1:])

            text_title = title.replace('\\n', ' ').replace('\\', ' ').strip()


            text_content = body.replace('\\n', ' ').replace('\\', ' ').strip()


            text_a = f'{text_title}.{text_content}{text_title} is a <mask>.'

            label = label_map[str(line[1])]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class TRECProcessor(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def read_data_for_create_examples(self, set_type: str):
        res_line = []
        res_line.append(['sentence', 'label'])

        if set_type == "test":
            with open(os.path.join(project_root, "data/trec/test.json")) as f:
                lines = f.readlines()

            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        if set_type == "train":
            with open(os.path.join(project_root, "data/trec/train.json")) as f:
                lines = f.readlines()
            # train_pool_labels = []
            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])
            #     train_pool_labels.append(i["_id"])
            # ALargs.class_labels = train_pool_labels

        if set_type == "dev":
            with open(os.path.join(project_root, "data/trec/dev.json")) as f:
                lines = f.readlines()

            for i in lines:
                i = json.loads(i)
                res_line.append([i["text"], str(i["_id"])])

        return res_line

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_data_for_create_examples(set_type='test'), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def get_label_map(self):
        label_map = {"0": "expression", "1": "entity", "2": "description", "3": "human", "4": "location", "5": "number"}

        return label_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        label_map = self.get_label_map()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"  # guid:'train-1'
            text_a = line[text_index]

            ### 根据最大长度进行截断：
            # raw_max_len = 256  - 18
            # text_a = text_a[:raw_max_len]
            text_a = '{} It was <mask>.'.format(text_a)
            text_a = text_a.replace('\\n', ' ').replace('\\', ' ').strip()
            label = label_map[str(line[1])]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


processors = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst2": Sst2Processor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "mpqa": MpqaProcessor,
    "mr": MrProcessor,
    "subj": SubjProcessor,
    "imdb": ImdbProcessor,
    "agnews": AgnewsProcessor,
    "yelp": YelpProcessor,
    "yahoo": YahooProcessor,
    "dbpedia": DBPediaProcessor,
    "trec": TRECProcessor,
}

output_modes = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst2": "classification",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "mpqa": "classification",
    "mr": "classification",
    "subj": "classification",

    ###Al
    "imdb": "classification",
    "agnews": "classification",
    "yelp": "classification",
    "yahoo": "classification",
    "dbpedia": "classification",
    "trec": "classification",

}

# # configuration names available in evaluate lib for glue
# # ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]
task_mappings_for_eval = {
    # 'sst2': 'sst2',
    # 'cola': 'cola',
    # 'mnli': 'mnli',
    # 'mnli-mm': 'mnli_mismatched',
    # 'qqp': 'qqp',
    # 'qnli': 'qnli',
    # 'rte': 'rte',
    # 'mrpc': 'mrpc',
    # 'mpqa': 'sst2',
    # 'mr': 'sst2',
    # 'subj': 'sst2',
    #
    # 'snli': 'qnli',

    ### AL
    'imdb': 'imdb',
    'agnews': 'agnews',
    'yelp': 'yelp',
    'yahoo': 'yahoo',
    'dbpedia': 'dbpedia',
    'trec': 'trec',

}

task_to_keys = {

    ###al
    "imdb": ("sentence1", None),
    "agnews": ("sentence1", None),
    "yelp": ("sentence1", None),
    "yahoo": ("sentence1", None),
    "dbpedia": ("sentence1", None),
    "trec": ("sentence1", None),
}

task_dir_mapping = {

    ### al
    "imdb": "IMDB",
    "agnews": "agnews",
    "yelp": "yelp",
    "yahoo": "yahoo",
    "dbpedia": "dbpedia",
    "trec": "trec",
}

