import json
import logging
import os
import random
from typing import List

import torch
from datasets import Value, load_dataset
from torch.utils.data import Dataset

from AL.query_strategy import find_examples
from ALconfig import ALargs
from utils.data_processors import convert_examples_to_features
from utils.data_processors import output_modes
from utils.data_processors import processors

logger = logging.getLogger(__name__)


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    # Get the list of labels from train split
    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")

    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]

    # Update the label list with the labels from validation and test set [If Different]
    for split in ["validation", "test"]:
        if split in raw_dataset:

            if isinstance(raw_dataset[split]["label"][0], list):
                val_or_test_labels = [label for sample in raw_dataset[split]["label"] for label in sample]
                val_or_test_labels = list(set(val_or_test_labels))
            else:
                val_or_test_labels = raw_dataset[split].unique("label")

            val_or_test_labels = [str(label) for label in val_or_test_labels]

            diff = set(val_or_test_labels).difference(set(label_list))
            if len(diff) > 0:
                # add the labels that appear in val/test but not in train, throw a warning
                logger.warning(
                    f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                )
                label_list += list(diff)

    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == '-1':
            logger.warning("Label -1 found in label list, removing it.")
            label_list.remove(label)

    label_list.sort()
    return label_list


def load_data(args):
    """
    See more about loading any type of standard or custom dataset at
    https://huggingface.co/docs/datasets/loading_datasets.html

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    """

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            path=args.dataset_path,
            name=args.dataset_name,
            cache_dir=args.cache_dir,
            token=args.token,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file

        # Get the test dataset: you can provide your own CSV/JSON test file
        if args.do_predict:
            if args.test_file is not None:
                train_extension = args.train_file.split(".")[-1]
                test_extension = args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=args.cache_dir,
                token=args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=args.cache_dir,
                token=args.token,
            )

    if args.remove_splits is not None:
        for split in args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if args.train_split_name is not None:
        logger.info(f"using {args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[args.train_split_name]
        raw_datasets.pop(args.train_split_name)

    if args.validation_split_name is not None:
        logger.info(f"using {args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[args.validation_split_name]
        raw_datasets.pop(args.validation_split_name)

    if args.test_split_name is not None:
        logger.info(f"using {args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[args.test_split_name]
        raw_datasets.pop(args.test_split_name)

    if args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if args.label_column_name is not None and args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(args.label_column_name, "label")

    return raw_datasets


def cast_regression_labels(raw_datasets):
    # regression requires float as label type, let's cast it if needed
    for split in raw_datasets.keys():
        if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
            logger.warning(
                f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
            )
            features = raw_datasets[split].features
            features.update({"label": Value("float32")})
            try:
                raw_datasets[split] = raw_datasets[split].cast(features)
            except TypeError as error:
                logger.error(
                    f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                )
                raise error
    return raw_datasets


def load_labels(raw_datasets, args):
    is_multi_label = False
    if args.dataset_name is not None:
        is_regression = args.dataset_name == "stsb"

        if not is_regression:
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            label_list = get_label_list(raw_datasets, split="train")
            num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")

        else:
            # Cast the labels to float if they are not already
            raw_datasets = cast_regression_labels(raw_datasets)
            num_labels = 1
            label_list = None
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # (Don't cast here since their type is being used to determine the task type)
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
            label_list = None

        else:
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = get_label_list(raw_datasets, split="train")
            num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")

    args.is_regression = is_regression
    args.num_labels = num_labels
    args.label_list = label_list
    args.is_multi_label = is_multi_label
    return


class PromptDataset_wEnc(Dataset):
    def __init__(self, args, task, tokenizer, enc_tokenizer, data_type="train", dynamic_pad=True):

        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.enc_tokenizer = enc_tokenizer
        self.data_type = data_type
        self.dynamic_pad = dynamic_pad

        # Get the features for the classifier model
        features = self.convert_to_features(
            model_name_or_path=self.args.model_name_or_path,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
        )


        # enc_features = self.convert_to_features(
        #     model_name_or_path=self.args.enc_model_name_or_path,
        #     tokenizer=self.enc_tokenizer,
        #     max_length=self.args.max_length,
        # )
        enc_features = features



        if data_type == "train":


            selected_features = [features[i] for i in ALargs.selected_index]
            enc_features = [enc_features[i] for i in ALargs.selected_index]

        ### AL  pool dev test
        else:
            selected_features = features
            enc_features = enc_features

        self.all_input_ids = [f.input_ids for f in selected_features]
        self.all_attention_mask = [f.attention_mask for f in selected_features]
        self.all_token_type_ids = [f.token_type_ids for f in selected_features]
        self.all_mask_pos = [f.mask_pos for f in selected_features]
        self.all_labels = [f.label for f in selected_features]


        self.all_enc_input_ids = [f.input_ids for f in enc_features]
        self.all_enc_attention_mask = [f.attention_mask for f in enc_features]
        self.all_enc_token_type_ids = [f.token_type_ids for f in enc_features]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):
        input_ids = self.all_input_ids[index]
        attention_mask = self.all_attention_mask[index]
        token_type_ids = self.all_token_type_ids[index]
        mask_pos = self.all_mask_pos[index]

        enc_input_ids = self.all_enc_input_ids[index]
        enc_attention_mask = self.all_enc_attention_mask[index]
        enc_token_type_ids = self.all_enc_token_type_ids[index]

        if self.all_labels is not None:
            label = self.all_labels[index]
            return {
                'enc_input_ids': enc_input_ids,
                'enc_attention_mask': enc_attention_mask,
                'enc_token_type_ids': enc_token_type_ids,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'mask_pos': mask_pos,
                'label': label,
            }
        else:
            return {
                'enc_input_ids': enc_input_ids,
                'enc_attention_mask': enc_attention_mask,
                'enc_token_type_ids': enc_token_type_ids,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'mask_pos': mask_pos,
            }

    def collate_fn(self, batch_data: List[dict]):

        batch_input_ids, batch_attention_mask = [], []
        batch_token_type_ids, batch_mask_pos, batch_labels = [], [], []
        batch_enc_input_ids, batch_enc_attention_mask = [], []
        batch_enc_token_type_ids = []

        all_length = [len(item['input_ids']) for item in batch_data]
        max_len = max(all_length) if self.dynamic_pad else self.args.max_length

        all_enc_length = [len(item['enc_input_ids']) for item in batch_data]
        max_enc_len = max(all_enc_length) if self.dynamic_pad else self.args.max_length

        for i, item in enumerate(batch_data):

            # # Latent Prompt Generator/Encoder model
            enc_input_ids = item['enc_input_ids']
            enc_input_ids += [self.enc_tokenizer.pad_token_id] * (max_enc_len - all_enc_length[i])
            batch_enc_input_ids.append(enc_input_ids)

            enc_attention_mask = item['enc_attention_mask']
            enc_attention_mask += [0] * (max_enc_len - all_enc_length[i])
            batch_enc_attention_mask.append(enc_attention_mask)

            enc_token_type_ids = item['enc_token_type_ids']
            enc_token_type_ids += [self.enc_tokenizer.pad_token_type_id] * (max_enc_len - all_enc_length[i])
            batch_enc_token_type_ids.append(enc_token_type_ids)

            # # Classifier model
            input_ids = item['input_ids']
            # input_ids += [self.tokenizer.pad_token_id] * (max_len - all_length[i])
            batch_input_ids.append(input_ids)

            attention_mask = item['attention_mask']
            # attention_mask += [0] * (max_len - all_length[i])
            batch_attention_mask.append(attention_mask)

            token_type_ids = item['token_type_ids']
            # token_type_ids += [self.tokenizer.pad_token_type_id] * (max_len - all_length[i])
            batch_token_type_ids.append(token_type_ids)

            mask_pos = item['mask_pos']
            batch_mask_pos.append(mask_pos)

            if self.all_labels is not None:
                label = item['label']
                batch_labels.append(label)

        batch_enc_input_ids = torch.tensor(batch_enc_input_ids, dtype=torch.long)
        batch_enc_attention_mask = torch.tensor(batch_enc_attention_mask, dtype=torch.long)
        batch_enc_token_type_ids = torch.tensor(batch_enc_token_type_ids, dtype=torch.long)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_mask_pos = torch.tensor(batch_mask_pos, dtype=torch.long)

        if len(batch_labels) != 0:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            # return (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_mask_pos, batch_labels)
            return {
                'enc_input_ids': batch_enc_input_ids,
                'enc_attention_mask': batch_enc_attention_mask,
                'enc_token_type_ids': batch_enc_token_type_ids,
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
                'token_type_ids': batch_token_type_ids,
                'mask_pos': batch_mask_pos,
                'labels': batch_labels,
            }
        else:
            return {
                'enc_input_ids': batch_enc_input_ids,
                'enc_attention_mask': batch_enc_attention_mask,
                'enc_token_type_ids': batch_enc_token_type_ids,
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
                'token_type_ids': batch_token_type_ids,
                'mask_pos': batch_mask_pos,
            }

    def convert_to_features(self, model_name_or_path, tokenizer, max_length):

        if self.args.local_rank not in [-1, 0] and self.data_type == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[self.task]()
        output_mode = output_modes[self.task]
        # Load data features from cache or dataset file

        data_type = self.data_type
        ####  无标签池和train使用一样的features
        if self.data_type == "pool":
            data_type = "train"
        if not self.args.ood:
            cached_features_file = os.path.join(
                self.args.data_dir,

                "cached_{}_{}_{}_{}".format(
                    data_type,
                    list(filter(None, model_name_or_path.split("/"))).pop(),
                    str(max_length),
                    str(self.task),
                ),
        )
        else: ## ood 按照ood数据集进行测试
            cached_features_file = os.path.join(
                self.args.data_dir,

                "cached_{}_{}_{}_{}_{}".format(
                    data_type,
                    list(filter(None, model_name_or_path.split("/"))).pop(),
                    str(max_length),
                    str(self.task),
                    str(self.args.ood_name),
                ),
            )
        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", self.args.data_dir)

            if self.data_type == "train" or self.data_type == "pool":
                examples = processor.get_train_examples(self.args.data_dir)
            elif self.data_type == "dev":
                examples = processor.get_dev_examples(self.args.data_dir)
            elif self.data_type == "test":
                # examples = processor.get_test_examples(self.args.data_dir,)
                examples = processor.get_test_examples(self.args)

            else:
                raise NotImplementedError

            label_map = processor.get_label_map()
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_map=label_map,
                max_length=max_length,
                output_mode=output_mode,
            )

            if self.args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.args.local_rank == 0 and not self.data_type == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        print(f"\n!!!!!!!!{data_type} features size:{len(features)}")
        logger.info(f"\n!!!!!!!!{data_type} features size:{len(features)}")
        return features
