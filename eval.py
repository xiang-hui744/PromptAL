import os
import time

import accelerate
import datasets
import evaluate
import torch
import transformers
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig

from ALconfig import ALargs
from utils.config import get_config
from utils.data import PromptDataset_wEnc as PromptDataset
from utils.data_processors import processors, output_modes, task_mappings_for_eval
from utils.model import IDPGSoftPromptGenerator, InstancePromptGenerator
from utils.modeling_roberta import RobertaForMaskedLM
from utils.xformer import load_tokenizer, get_huggingface_path, load_base_model


def load_encoder(args, logger, accelerator):
    """
            Initialize the encoder.
    """
    if args.peft_method == 'idpg':
        model = IDPGSoftPromptGenerator(
            args=args,
        )
    elif args.peft_method == 'lopa':
        model = InstancePromptGenerator(args)
    else:
        return None

    # Load the model state dict on the CPU to avoid an OOM error.
    loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
    loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
    if args.peft_method == 'idpg':
        loaded_state_dict = {k: v for k, v in loaded_state_dict.items() if 'base' not in k}  # Remove base model weights
        model.load_state_dict(loaded_state_dict,
                              strict=False)  # strict=False allows for partial loading [IDPG-specific]
    else:
        model.load_state_dict(loaded_state_dict, strict=True)

    # release memory
    del loaded_state_dict

    # Log the loaded checkpoint
    msg = "[INFO] Loaded encoder checkpoint from path: {}".format(args.clf_predictor_path)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    return model


def load_pt(args, logger, accelerator, model):
    from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftModel

    if not os.path.exists(args.load_adapter_from):
        logger.error("Please specify the correct path to load the model adapters from")
        raise ValueError("Please specify the correct path to load the model adapters from")

    # Get the config
    peft_config = PromptTuningConfig(
        task_type=TaskType.MASKED_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=args.num_virtual_tokens,
    )

    # Load the model adapters - in place
    model = PeftModel.from_pretrained(
        model=model,
        model_id=args.load_adapter_from,  # Must be a directory containing the model files
        config=peft_config,
    )
    msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    # This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
    args.total_virtual_tokens = args.num_virtual_tokens * peft_config.num_transformer_submodules
    args.word_embedding_dim = peft_config.token_dim
    return model


def load_lora(args, logger, accelerator, model):
    from peft import PeftModel

    if not os.path.exists(args.load_adapter_from):
        logger.error("Please specify the correct path to load the model adapters from")
        raise ValueError("Please specify the correct path to load the model adapters from")

    # # Load the model adapters - in place
    model = PeftModel.from_pretrained(
        model=model,
        model_id=args.load_adapter_from,  # Must be a directory containing the model files
    )

    # merge the adapter weights with the base model. doesn’t keep the adapter weights in memory.
    model.merge_and_unload()

    msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    return model


def load_idpg(args, logger, accelerator, model):
    from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftIDPGModel

    if not os.path.exists(args.load_adapter_from):
        logger.error("Please specify the correct path to load the model adapters from")
        raise ValueError("Please specify the correct path to load the model adapters from")

    # Get the config
    peft_config = PromptTuningConfig(
        task_type=TaskType.IDPG_MASKED_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=args.num_virtual_tokens,
    )

    # # Load the model adapters - in place
    model = PeftIDPGModel.from_pretrained(
        model=model,
        model_id=args.load_adapter_from,  # Must be a directory containing the model files
        config=peft_config,
    )

    msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    # This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
    args.total_virtual_tokens = args.num_virtual_tokens * peft_config.num_transformer_submodules
    args.word_embedding_dim = peft_config.token_dim

    return model


def load_dept(args, logger, accelerator, model):
    from custom_peft import PromptTuningLoRAConfig, TaskType, PromptTuningInit, PeftDEPTModel
    if not os.path.exists(args.load_adapter_from):
        logger.error("Please specify the correct path to load the model adapters from")
        raise ValueError("Please specify the correct path to load the model adapters from")

    dept_config = PromptTuningLoRAConfig(
        task_type=TaskType.DEPT_MASKED_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=5,  # Set to 5 for DEPT with which number of params match PT(m=10)
        tokenizer_name_or_path=args.tokenizer_name,
        r=4,  # LoRA parameter in the paper
        token_dim=model.config.hidden_size,
        max_length=args.max_length,
        save_lora_embeddings=True,
    )

    model = PeftDEPTModel.from_pretrained(
        model=model,
        model_id=args.load_adapter_from,  # Must be a directory containing the model files
        config=dept_config,
    )

    msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    args.total_virtual_tokens = 5 * dept_config.num_transformer_submodules
    args.word_embedding_dim = dept_config.token_dim

    return model


def load_lopa(args, logger, accelerator, model):
    from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftLopaModel
    if not os.path.exists(args.load_adapter_from):
        logger.error("Please specify the correct path to load the model adapters from")
        raise ValueError("Please specify the correct path to load the model adapters from")

    # Get the config
    peft_config = PromptTuningConfig(
        task_type=TaskType.LOPA_MASKED_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=args.num_virtual_tokens,
    )

    # # Load the model adapters - in place
    model = PeftLopaModel.from_pretrained(
        model=model,
        model_id=args.load_adapter_from,  # Must be a directory containing the model files
        config=peft_config,
    )

    msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
    logger.info(msg)
    if accelerator.is_local_main_process:
        print(msg)

    # This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
    args.total_virtual_tokens = args.num_virtual_tokens * peft_config.num_transformer_submodules
    args.word_embedding_dim = peft_config.token_dim
    return model


def load_foundation_model(args, logger, accelerator):
    # Get the foundation model
    config, model = load_base_model(
        args,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        model_class=RobertaForMaskedLM,  # This is my custom class not the one from transformers
        config_class=RobertaConfig
    )

    # [FFT] If the single checkpoint path is provided, load the checkpoint
    if args.load_base_from_path is not None:
        # We load the model state dict on the CPU to avoid an OOM error.
        loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        model.load_state_dict(loaded_state_dict, strict=True)

        # release memory
        del loaded_state_dict

        # Log the loaded checkpoint
        message = "[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path)
        logger.info(message)
        if accelerator.is_local_main_process:
            print(message)

    # [For PEFT methods]
    if args.peft_method == 'pt':
        model = load_pt(args, logger, accelerator, model)
        config.total_virtual_tokens = args.total_virtual_tokens

    elif args.peft_method == 'lora':
        model = load_lora(args, logger, accelerator, model)

    elif args.peft_method == 'idpg':
        model = load_idpg(args, logger, accelerator, model)
        config.total_virtual_tokens = args.total_virtual_tokens

    elif args.peft_method == 'lopa':
        model = load_lopa(args, logger, accelerator, model)
        config.total_virtual_tokens = args.total_virtual_tokens

    elif args.peft_method == 'fft':
        pass

    else:
        raise NotImplementedError(f"PEFT method = {args.peft_method} not implemented")

    # ALargs.foundation_model = model

    return config, model


class Evaluator(object):

    def __init__(self, args, logger):
        self.args = args

        # init with accelerate
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        with self.accelerator.main_process_first():
            self.logger = logger

        self.logger.info("Accelerator State:\n")
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New Evaluation process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.info(f"Experiment name: {args.project_name}")
        self.logger.info(f"Experiment directory: {self.args.log_dir}")

        # setup tokenizer
        logger.info(f"[INFO] Loading Foundation Model's tokenizer from {get_huggingface_path(args.model_type)}")
        self.tokenizer = load_tokenizer(args, args.model_type, args.tokenizer_name)
        logger.info(
            f"[INFO] Loading Soft (Latent) Prompt Generator's tokenizer from {get_huggingface_path(args.enc_model_type)}")
        self.lp_gen_tokenizer = load_tokenizer(args, args.enc_model_type, get_huggingface_path(args.enc_model_type))

        # prepare glue task
        self.prepare_glue_task()

        # setup model
        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building model...")
            start = time.monotonic_ns()
            self.config, self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.model)
            self.logger.info(f"[INFO] Building model done in {(end - start) / 1e6:.2f}ms")

        # Get data
        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building dataset...")
            start = time.monotonic_ns()
            self.eval_dataset = self.get_data()
            end = time.monotonic_ns()
            self.logger.info(f"[INFO] Building dataset done in {(end - start) / 1e6:.2f}ms")

        # Load data Loaders
        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building dataloader...")
            start = time.monotonic_ns()
            self.eval_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"[INFO] Building dataloader done in {(end - start) / 1e6:.2f}ms")

        # accelerate prepare
        self.logger.info("[INFO] Initializing accelerate...")
        start = time.monotonic_ns()
        self._accelerator_prepare()
        end = time.monotonic_ns()
        self.logger.info(f"[INFO] Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # Setup the evaluation
        self.label_ids = self.setup_eval()
        # Get the metric function
        evaluate_path = ''
        accuracy_path = os.path.join(evaluate_path, "metrics/accuracy")
        f1_path = os.path.join(evaluate_path, "metrics/f1")
        recall_path = os.path.join(evaluate_path, "metrics/recall")


        glue_path = os.path.join(evaluate_path, "metrics/glue")
        if self.args.dataset_name is not None:

            self.metric = evaluate.load(accuracy_path)
            self.metric = evaluate.load(f1_path)
            # self.metric = evaluate.load(recall_path)

        else:

            self.metric = evaluate.load(accuracy_path)

        # save config file path
        self.config_save_path = os.path.join(self.args.log_dir, "args.json")
        self.args.device = self.accelerator.device

    def prepare_glue_task(self):
        task_name = self.args.dataset_name
        processor = processors[task_name]()
        self.args.output_mode = output_modes[task_name]
        self.args.is_regression = self.args.output_mode == "regression"
        self.args.is_multi_label = self.args.output_mode == "multilabel_classification"
        self.args.label_list = processor.get_labels()
        self.args.num_labels = len(self.args.label_list)

    def _init_accelerator(self):

        project_config = ProjectConfiguration(
            logging_dir=self.args.log_dir,
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # when using DeepSpeed, the `gradient_accumulation_steps` is properly set either
        # > from the DeepSpeed plugin/config
        # > from `accelerate launch` via `--gradient_accumulation_steps`
        # > defaulting to the passed `args.gradient_accumulation_steps` (using this + setting auto in the config file)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            project_config=project_config,
            kwargs_handlers=[kwargs],
        )

    def get_data(self):
        dataset = PromptDataset(
            self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer,
            data_type='test',
            dynamic_pad=self.args.dynamic_pad
        )
        return dataset

    def _build_dataloader(self):
        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_dataset.collate_fn
        )
        return eval_dataloader

    def _build_model(self):
        # Load the foundation model
        fm_config, foundation_model = load_foundation_model(self.args, self.logger, self.accelerator)

        # Load the encoder
        encoder = load_encoder(self.args, self.logger, self.accelerator)

        if self.args.peft_method == 'lopa':
            from utils.model import LOPA
            model = LOPA(fm_config, encoder, foundation_model)
        elif self.args.peft_method == 'idpg':
            from utils.model import IDPG
            model = IDPG(fm_config, encoder, foundation_model)
        elif self.args.peft_method == 'pt' or self.args.peft_method == 'dept':
            from utils.model import DummyModel
            model = DummyModel(fm_config, foundation_model)
        else:
            model = foundation_model

        return fm_config, model

    def _accelerator_prepare(self):

        self.eval_dataloader, self.model = self.accelerator.prepare(self.eval_dataloader, self.model)

    def setup_eval(self):
        processor = processors[self.args.dataset_name]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = self.tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])

        # if self.accelerator.is_main_process:
        # 	print("[DEBUG] Label IDs: ", label_ids)
        return label_ids

    def eval(self):
        self.model.eval()

        samples_seen = 0
        for step, batch in tqdm(
                enumerate(self.eval_dataloader),
                desc=f"Evaluating",
                unit="batch",
                colour="GREEN",
                leave=False,
                dynamic_ncols=True,
                smoothing=0.04,
                disable=not self.accelerator.is_main_process,
                total=len(self.eval_dataloader),
        ):
            with torch.no_grad():
                if self.args.peft_method in ['idpg', 'lopa', 'pt', 'dept']:
                    output = self.model(batch)
                else:
                    output = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        mask_pos=batch['mask_pos'],
                        labels=batch['labels']
                    )
                logits = output.logits
                # breakpoint()
                # Logits for label ids
                logits = logits[:, self.label_ids]

            logits, references = self.accelerator.gather((logits, batch["labels"]))

            # Update the label ids in the references to 0, 1, ...
            for i, label in enumerate(self.label_ids):
                references[references == label] = i
            # Get the predictions
            predictions = logits.argmax(dim=-1) if not self.args.is_regression else logits.squeeze()

            # If we are in a multiprocess environment, the last batch has duplicates
            if self.accelerator.num_processes > 1:
                if step == len(self.eval_dataloader) - 1:
                    predictions = predictions[: len(self.eval_dataloader.dataset) - samples_seen]
                    references = references[: len(self.eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
#predictions和references 是tensor的010101
            self.metric.add_batch(
                predictions=predictions,
                references=references,
            )

        # Compute the final evaluation metric
        eval_metric = self.metric.compute()
        for key, metric in eval_metric.items():
            self.logger.info("  |- Eval/{}: {:.6f}".format(key, metric))
            print("  |- Eval/{}: {:.6f}".format(key, metric))

    def predict(self, force_valid_labels=True):
        """
        Predict with the model
        :param force_valid_labels: If True, the predictions will be forced to be in the valid label set
        :return:
        """
        self.model.eval()

        samples_seen = 0
        for step, batch in tqdm(
                enumerate(self.eval_dataloader),
                desc=f"Predicting",
                unit="batch",
                colour="BLUE",
                leave=False,
                dynamic_ncols=True,
                smoothing=0.04,
                disable=not self.accelerator.is_main_process,
                total=len(self.eval_dataloader),
        ):
            with torch.no_grad():
                batch['labels'] = None
                if self.args.peft_method in ['idpg', 'lopa', 'pt', 'dept']:
                    output = self.model(batch)
                else:
                    output = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        mask_pos=batch['mask_pos'],
                        labels=batch['labels']
                    )
                logits = output.logits

            inputs, logits = self.accelerator.gather((batch['input_ids'], logits))

            # Get the prediction which is the idx from label ids with the max
            if force_valid_labels:
                logits = logits[:, self.label_ids]
                predictions = logits.argmax(dim=-1) if not self.args.is_regression else logits.squeeze()
                predictions = [self.label_ids[pred] for pred in predictions]

            # Get the predictions [Unforced]
            else:
                predictions = logits.argmax(dim=-1) if not self.args.is_regression else logits.squeeze()
                predictions = predictions.tolist()

            # If we are in a multiprocess environment, the last batch has duplicates
            if self.accelerator.num_processes > 1:
                if step == len(self.eval_dataloader) - 1:
                    predictions = predictions[: len(self.eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += len(predictions)

            # Save the input and corresponding predictions, separated by a | character
            predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
            with open(os.path.join(self.args.log_dir, "predictions.txt"), "a") as f:
                for inp, pred in zip(inputs, predictions):
                    f.write(f"{inp} | {pred}\n")


def main():
    args, logger = get_config()
    logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.

    # Initialize the evaluator
    evaluator = Evaluator(args, logger)

    # Evaluate the model. [IMP] this needs G.T. labels
    if args.do_evaluation:
        evaluator.eval()

    # Predict with the model
    if args.do_prediction:
        evaluator.predict(force_valid_labels=args.force_prediction_in_valid_labels)


if __name__ == '__main__':
    main()
