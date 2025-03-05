import math
import os
import time
from collections import defaultdict
import copy
import accelerate
import datasets
import evaluate
import torch
import transformers
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from wandb.integration.sklearn.plot.classifier import calibration_curve

from AL.get_embedding import get_hmask, get_cls, get_simcse
from utils.data import PromptDataset_wEnc as PromptDataset
from utils.data_processors import processors, output_modes, task_mappings_for_eval
from utils.xformer import load_tokenizer, get_huggingface_path

from ALconfig import ALargs


class BaseTrainer(object):

    def __init__(self, args, logger):
        self.args = args

        self.eval_best_model = None
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
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.info(f"Experiment name: {args.project_name}")
        self.logger.info(f"Experiment directory: {self.args.log_dir}")

        # init counts
        self.step: int = 0
        self.epoch: int = 0

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

            # Get the number of trainable parameters
            lp_gen_trainable_params, lp_gen_all_params, fm_trainable_params, fm_all_params = self.count_parameters()
            if lp_gen_all_params is not None and lp_gen_trainable_params is not None:
                msg = (
                    f"\nSoft (Latent) Prompt Generator: trainable params: {lp_gen_trainable_params:,d} || all params: {lp_gen_all_params:,d} ||"
                    f" trainable%: {100 * lp_gen_trainable_params / lp_gen_all_params}")
                self.logger.info(msg)
                print(msg)
            if fm_all_params is not None and fm_trainable_params is not None:
                msg = (
                    f"\nFoundation Model: trainable params: {fm_trainable_params:,d} || all params: {fm_all_params:,d} ||"
                    f" trainable%: {100 * fm_trainable_params / fm_all_params}")
                self.logger.info(msg)
                print(msg)

        #### !!!! Get pool data loader

        pool_dataset = self.get_pool_data()
        self.pool_dataset = pool_dataset
        print('!!! 构造pool_dataset,大小为：', len(pool_dataset))
        ALargs.unlabeled_size = len(pool_dataset)

        # ALargs.pool_dataloader = self.build_dataloader(task='pool')
        self.build_dataloader(task='pool')

        print('!!! 构造pool_dataloader')

        #### !!!!Get dataset
        ###  !!! train dataset需要隔离出去 初始化时没有训练集，ALargs选样本以后加入训练集
        self.train_dataset = None

        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building test and eval dataset...")
            print("[INFO] Building test and eval dataset...")
            start = time.monotonic_ns()

            ######### ALargs 利用 pool_dataloader 和 model 进行各种AL的样本选择
            ######### train_dataset是样本选择后的数据集

            # self.train_dataset, self.eval_dataset, self.test_dataset = self.get_data()
            self.eval_dataset, self.test_dataset = self.get_data()

            end = time.monotonic_ns()
            self.logger.info(f"[INFO] Building test and eval dataset done in {(end - start) / 1e6:.2f}ms")

        ###  Load test and eval data Loaders
        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building dataloader...")
            start = time.monotonic_ns()
            # self.train_dataloader, self.eval_dataloader, self.test_dataloader = self._build_dataloader()
            self.eval_dataloader, self.test_dataloader = self._build_dataloader()

            end = time.monotonic_ns()
            print("!!!! 构造了本次AL的test、dev dataloader\n")
            self.logger.info("!!!! 构造了本次AL的test、dev dataloader\n")
            self.logger.info(f"[INFO] Building dataloader done in {(end - start) / 1e6:.2f}ms")

        # # optimizer & scheduler
        # with self.accelerator.main_process_first():
        #     self.logger.info("[INFO] Building optimizer and scheduler...")
        #     start = time.monotonic_ns()
        #     self.optimizer = self._build_optimizer()
        #     self.scheduler = self._build_scheduler()
        #     end = time.monotonic_ns()
        #     self.logger.info(
        #         f"[INFO] Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
        #     )
        #
        # accelerate prepare
        self.logger.info("[INFO] Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator_prepare_model_pool_test_dev_dataloader()
        end = time.monotonic_ns()
        self.logger.info(f"[INFO] Initializing accelerate done in {(end - start) / 1e6:.2f}ms")
        #
        # # We need to recalculate our total training steps as the size of the training dataloader may have changed after
        # # Accelerator's prepare function.
        # self.recalculate_training_metrics()

        # Setup the evaluation
        self.label_ids = self.setup_eval()
        ALargs.label_ids = self.label_ids

        evaluate_path = ''
        accuracy_path = os.path.join(evaluate_path, "metrics/accuracy")
        f1_path = os.path.join(evaluate_path, "metrics/f1")
        glue_path = os.path.join(evaluate_path, "metrics/glue")


        # if self.args.dataset_name in ["yelp"]:
        #     ### 多分类的f1：
        #     self.f1_metric = evaluate.load(f1_path,average='macro')
        # else:
        self.f1_metric = evaluate.load(f1_path)



        if self.args.dataset_name in ["imdb", "agnews","yelp","agnews","yahoo"]:
            self.metric = evaluate.load(accuracy_path)
        # elif self.args.dataset_name is not None:
        #     self.metric = evaluate.load(glue_path, task_mappings_for_eval[self.args.dataset_name])
        else:
            self.metric = evaluate.load(accuracy_path)

        # save config file path
        self.config_save_path = os.path.join(self.args.log_dir, "args.json")
        self.args.device = self.accelerator.device

        # Finally, initialize the trackers. During init of the model we computed new arguments. Thus setting after that.
        self.init_trackers()

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
        if self.args.wandb_logging:
            self.accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                log_with=["wandb"],
                project_config=project_config,
                kwargs_handlers=[kwargs],
            )
        else:
            self.accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                project_config=project_config,
                kwargs_handlers=[kwargs],
            )

    def get_data(self):
        # train_dataset = PromptDataset(
        #     self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer, data_type='train',
        #     dynamic_pad=self.args.dynamic_pad
        # )
        eval_dataset = PromptDataset(
            self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer, data_type='dev',
            dynamic_pad=self.args.dynamic_pad
        )
        test_dataset = PromptDataset(
            self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer, data_type='test',
            dynamic_pad=self.args.dynamic_pad
        )

        # return train_dataset, eval_dataset, test_dataset
        return eval_dataset, test_dataset

    def get_train_dataset(self):

        train_dataset = PromptDataset(
            self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer, data_type='train',
            dynamic_pad=self.args.dynamic_pad
        )
        self.train_dataset = train_dataset
        return train_dataset

    def get_pool_data(self):
        pool_dataset = PromptDataset(
            self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer, data_type='pool',
            dynamic_pad=self.args.dynamic_pad
        )

        return pool_dataset

    def get_optimizer(self):
        # optimizer & scheduler
        with self.accelerator.main_process_first():
            self.logger.info("[INFO] Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            self.logger.info(
                f"[INFO] Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # accelerate prepare  和GPU相关
        self.logger.info("[INFO] Initializing train accelerate...")
        start = time.monotonic_ns()
        self._accelerator_prepare()
        end = time.monotonic_ns()
        self.logger.info(f"[INFO] Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # We need to recalculate our total training steps as the size of the training dataloader may have changed after
        # Accelerator's prepare function.
        self.recalculate_training_metrics()

    def _build_dataloader(self):
        # train_sampler = RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(
        #     self.train_dataset)
        # train_dataloader = DataLoader(
        #     self.train_dataset,
        #     sampler=train_sampler,
        #     batch_size=self.args.per_device_train_batch_size,
        #     collate_fn=self.train_dataset.collate_fn
        # )

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_dataset.collate_fn
        )

        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.accelerator.gradient_accumulation_steps)
        # self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch

        test_sampler = SequentialSampler(self.test_dataset)
        test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.args.per_device_test_batch_size,
            collate_fn=self.test_dataset.collate_fn
        )

        # return train_dataloader, eval_dataloader

        # return train_dataloader, eval_dataloader, test_dataloader

        return eval_dataloader, test_dataloader

    ### AL 为pool构造dataloader
    def build_dataloader(self, task='pool'):

        dataloader = None
        if task == 'train':
            sampler = SequentialSampler(self.train_dataset)
            dataloader = DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.train_dataset.collate_fn
            )
            num_update_steps_per_epoch = math.ceil(len(dataloader) / self.accelerator.gradient_accumulation_steps)
            self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
            self.train_dataloader = dataloader

        if task == 'pool':
            sampler = SequentialSampler(self.pool_dataset)

            dataloader = DataLoader(
                self.pool_dataset,
                sampler=sampler,
                batch_size=self.args.per_device_test_batch_size,
                collate_fn=self.pool_dataset.collate_fn
            )
            self.pool_dataloader = dataloader

    def _build_model(self):
        raise NotImplementedError

    def _build_optimizer(self):

        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
        optimizer_cls = (
            torch.optim.AdamW
            if self.accelerator.state.deepspeed_plugin is None
               or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else accelerate.utils.DummyOptim
        )
        optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.args.lr)

        return optimizer

    def _build_scheduler(self):

        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
                self.accelerator.state.deepspeed_plugin is None
                or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=int(0.06 * self.args.max_train_steps),
                num_training_steps=self.args.max_train_steps,
            )
        else:
            lr_scheduler = accelerate.utils.DummyScheduler(
                self.optimizer,
                total_num_steps=self.args.max_train_steps,
                warmup_num_steps=int(0.06 * self.args.max_train_steps),
            )
        return lr_scheduler

    def _accelerator_prepare(self):

        # self.train_dataloader, self.eval_dataloader, self.test_dataloader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
        #     self.train_dataloader, self.eval_dataloader, self.test_dataloader, self.model, self.optimizer,
        #     self.scheduler)
        self.train_dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.train_dataloader, self.optimizer, self.scheduler)

    def accelerator_prepare_model_pool_test_dev_dataloader(self):
        self.pool_dataloader, self.eval_dataloader, self.test_dataloader, self.model = self.accelerator.prepare(
            self.pool_dataloader, self.eval_dataloader, self.test_dataloader, self.model)

    def recalculate_training_metrics(self):

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps)
        self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch

        # # After wards we recalculate our number of training epochs.
        # Keep this. Useful when max_train_steps is to be set manually
        self.args.num_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        self.args.total_batch_size = (
                self.args.per_device_train_batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
        )

        self.logger.info("\n")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.logger.info("\n")

    def init_trackers(self):
        # Initialize the trackers
        raise NotImplementedError

    def count_parameters(self):
        raise NotImplementedError

    def forward(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """
        raise NotImplementedError

    def _train_step(self, batch):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """

        with self.accelerator.accumulate(self.model):
            output = self.forward(batch)

            # Classification function = -log(p(x|z))
            clf_loss = output.loss

            # Total loss
            total_loss = clf_loss

            # BP and Grad Updated
            self.accelerator.backward(total_loss)
            # Compute the gradients norm
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                # Updating the current step under the accumulate context manager takes care of everything
                self.step += 1

        return {
            f"total_loss": total_loss.detach().cpu().numpy().item(),
            f"clf_loss": clf_loss.detach().cpu().numpy().item(),
        }

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
                one epoch. See ``train_loop`` for usage.
        """

        # Set the model to train mode
        self.model.train()

        train_metrics: dict = {}

        for batch in tqdm(
                self.train_dataloader,
                desc=f"Training Epoch {self.epoch}",
                unit="batch",
                colour="GREEN",
                leave=False,
                dynamic_ncols=True,
                smoothing=0.04,
                disable=not self.accelerator.is_main_process,
        ):
            train_losses = self._train_step(batch)

            for key, value in train_losses.items():
                if key not in train_metrics.keys():
                    train_metrics[key] = value
                else:
                    train_metrics[key] += value

            if self.args.wandb_logging:
                self.accelerator.log(
                    {
                        "Step/Total Loss": train_losses["total_loss"],
                        "Step/Classification Loss": train_losses["clf_loss"],
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )

        # self.accelerator.wait_for_everyone()

        # Compute the average losses for the epoch
        for key in train_metrics.keys():
            train_metrics[key] = (
                    train_metrics[key] / len(self.train_dataloader) * self.args.gradient_accumulation_steps
            )

        return train_metrics

    def setup_eval(self):
        processor = processors[self.args.dataset_name]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = self.tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])

        if self.accelerator.is_main_process:
            print("[DEBUG] Label IDs: ", label_ids)
        return label_ids

    def _eval_epoch(self, dataloader, color="YELLOW"):
        self.model.eval()

        samples_seen = 0
        loss = torch.tensor(0.,device=self.accelerator.device)
        # print('\n 开始dev——————')
        for step, batch in tqdm(
                enumerate(dataloader),
                desc=f"第{ALargs.current_iterations}次迭代 Evaluating Epoch {self.epoch}",
                unit="batch",
                colour=color,
                leave=False,
                dynamic_ncols=True,
                smoothing=0.04,
                disable=not self.accelerator.is_main_process,
                total=len(dataloader),
        ):
            with torch.no_grad():
                outputs = self.forward(batch)
                logits = outputs.logits

                #### AL 保存dev loss最小的best模型方便后续进行测试
                loss=loss+outputs.loss

                # breakpoint()
                # Logits for label ids
                logits = logits[:, self.label_ids]

            reference_label = batch["labels"]
            logits, references = self.accelerator.gather((logits, batch["labels"]))


            # Update the label ids in the references to 0, 1, ...
            for i, label in enumerate(self.label_ids):
                references[references == label] = i

            a = torch.equal(reference_label, references)

            ### AL 校准

            predictions = logits.argmax(dim=-1) if not self.args.is_regression else logits.squeeze()

            # If we are in a multiprocess environment, the last batch has duplicates
            if self.accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    predictions = predictions[: len(dataloader.dataset) - samples_seen]
                    references = references[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            self.metric.add_batch(
                predictions=predictions,
                references=references,
            )
            self.f1_metric.add_batch(
                predictions=predictions,
                references=references,
            )

        # Compute the final evaluation metric
        acc_metric = self.metric.compute()
        if self.args.dataset_name in ["yelp","agnews","dbpedia","trec","yahoo"]:
            ### 多分类的f1：
            f1_metric = self.f1_metric.compute(average='macro')
        else:
            f1_metric = self.f1_metric.compute()

        return acc_metric, f1_metric, loss/len(dataloader.dataset)

    def predict_pool(self):
        """

        Returns: 使用yield逐批次返回无标签池的logits

        """
        self.model.eval()

        for step, batch in tqdm(
                enumerate(self.pool_dataloader),
                desc=f"Evaluating Pool ",
                unit="batch",
                colour="BLUE",
                leave=False,
                dynamic_ncols=True,
                smoothing=0.04,
                disable=not self.accelerator.is_main_process,
                total=len(self.pool_dataloader),
        ):
            with torch.no_grad():
                outputs = self.forward(batch)
                logits = outputs.logits




            # reference_label = batch["labels"]
            logits = self.accelerator.gather((logits))

            #### mask_pos
            if ALargs.pooling and ALargs.vector != 'simcse' and 'hmask' in ALargs.AL_method:
                if ALargs.vector == 'hmask':
                    batch_hmask = get_hmask(last_hidden_state=outputs.hidden_states[-1], mask_indices=batch['mask_pos'])
                if ALargs.vector == 'cls':
                    batch_hmask = get_cls(last_hidden_state=outputs.hidden_states[-1])

                combined_tensor = torch.cat((ALargs.h_mask, batch_hmask), dim=0)
                ALargs.h_mask = combined_tensor


            # 使用yield逐批次返回logits:batch*词表
            yield logits

    def train_loop(self):
        r"""Training loop. The public entry of training process."""

        best_metrics = defaultdict(lambda: 0)

        inf_value = float('inf')
        best_eval_loss = torch.tensor(inf_value,device=self.accelerator.device)

        # Do evaluation epoch at the start
        self.logger.info("\n")
        self.logger.info("-" * 32)
        self.logger.info("Epoch {}: ".format(-1))

        print("\n")
        print("-" * 32)
        print("Epoch {}: ".format(-1))

        eval_metrics, _,eval0_loss = self._eval_epoch(dataloader=self.eval_dataloader)

        best_overall = sum(eval_metrics.values()) / len(eval_metrics)
        for key, metric in eval_metrics.items():
            best_metrics[key] = metric
            self.logger.info("  |- Eval/{}: {:.6f}".format(key, metric))
            print("  |- Eval/{}: {:.6f}".format(key, metric))
        # self.accelerator.wait_for_everyone()
        while self.epoch < self.args.num_epochs:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))
            print("\n")
            print("-" * 32)
            print("Epoch {}: ".format(self.epoch))

            # Do training epoch
            train_metrics = self._train_epoch()

            # Do evaluation epoch
            eval_metrics, _ , eval_loss = self._eval_epoch(dataloader=self.eval_dataloader)

            # Update the best overall
            overall = sum(eval_metrics.values()) / len(eval_metrics)
            if overall > best_overall:
                best_overall = overall
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print(f"\n!!!! eval loss 变小为{best_eval_loss} ")
                self.logger.info(f"\n!!!! eval loss 变小为{best_eval_loss} ")
                print("!!!! 最佳模型更新")
                #### 进行测试
                if self.epoch>7:
                    self.logger.info(f"\n\n！！！epoch {self.epoch} Test Begin!!!!! ")
                    print(f"！！！epoch {self.epoch} Test Begin!!!!!  ")

                    acc_metric, f1_metric, _ = self._eval_epoch(dataloader=self.test_dataloader, color="red")
                    for key, metric in acc_metric.items():
                        self.logger.info("  |- Test/{}: {:.6f}".format(key, metric))
                        print("\n  |- Test/{}: {:.6f}".format(key, metric))
                        ALargs.test_results[f'{ALargs.current_iterations}_{self.epoch}_acc'].append(metric)

                    for key, metric2 in f1_metric.items():
                        self.logger.info("  |- Test/{}: {:.6f}".format(key, metric2))
                        print("\n  |- Test/{}: {:.6f}".format(key, metric2))
                        ALargs.test_results[f'{ALargs.current_iterations}_{self.epoch}_f1'].append(metric2)
                    print('\n\n')


                # Log the metrics
            for key, metric in train_metrics.items():
                self.logger.info("  |- Train/{}: {:.6f}".format(key, metric))
                print("  |- Train/{}: {:.6f}".format(key, metric))
                if self.args.wandb_logging:
                    self.accelerator.log({"Epoch/{}".format(key): metric}, step=self.step)
                    print({"Epoch/{}".format(key): metric})

            for key, metric in eval_metrics.items():
                self.logger.info("  |- Eval/{}: {:.6f}".format(key, metric))
                print("  |- Eval/{}: {:.6f}".format(key, metric))
                if self.args.wandb_logging:
                    self.accelerator.log({"Epoch/{}".format(key): metric}, step=self.step)
                    print({"Epoch/{}".format(key): metric})

                # Tracking the best metrics
                if key in best_metrics.keys():
                    if metric > best_metrics[key]:
                        best_metrics[key] = metric
                        # self.accelerator.wait_for_everyone()
                        # if self.accelerator.is_main_process:
                        self.save(f"best_{key}")
                else:
                    best_metrics[key] = metric
                    # self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self.save(f"best_{key}")

            # Update info for each epoch
            self.epoch += 1

            if self.args.save_every > 0 and self.epoch % self.args.save_every == 0:
                # self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.save(f"epoch_{self.epoch}")

        ### 每次模型训练完加一个测试结果
        self.logger.info("！！！Test Begin!!!!! ")
        print("！！！Test Begin!!!!!  ")

        acc_metric, f1_metric,_ = self._eval_epoch(dataloader=self.test_dataloader, color="red")
        for key, metric in acc_metric.items():
            self.logger.info("  |- Test/{}: {:.6f}".format(key, metric))
            print("\n  |- Test/{}: {:.6f}".format(key, metric))
            ALargs.test_results[f'{ALargs.current_iterations}_{self.epoch}_acc'].append(metric)

        for key, metric2 in f1_metric.items():
            self.logger.info("  |- Test/{}: {:.6f}".format(key, metric2))
            print("\n  |- Test/{}: {:.6f}".format(key, metric2))
            ALargs.test_results[f'{ALargs.current_iterations}_{self.epoch}_f1'].append(metric2)



        # Finish training and save final checkpoint
        # self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # self.accelerator.save_state(os.path.join(self.args.log_dir, "final_epoch"))
            self.save("final")

        self.accelerator.end_training()

        self.logger.info("\n")
        self.logger.info("=" * 32)
        self.logger.info("Training done.")
        self.logger.info("=" * 32)
        self.logger.info("Best overall performance so far : {:.6f}".format(best_overall))

    def save(self, dir_tag: str):

        raise NotImplementedError
