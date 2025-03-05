import argparse
import json
import logging
import os
import random
from collections import OrderedDict
from datetime import datetime

from transformers import set_seed
from transformers.utils.versions import require_version

from ALconfig import ALargs, class_labels
from utils.custom import is_rank_0, create_dir, log_dist, set_dist
# from utils.data_processors import processors, task_dir_mapping
from utils.xformer import get_huggingface_path

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# from transformers.utils import check_min_version
# check_min_version("4.40.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

args = None
def get_config():
    # Define the parameters
    # model_type = "roberta-large"
    # enc_model_type = "roberta-large"

    model_type = "roberta-base"
    enc_model_type = "roberta-base"

    # model_type = "bert-base-uncased"
    # enc_model_type = "bert-base-uncased"

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument("--peft_method", type=str, default='lopa',
                        choices=['lopa', 'pt', 'idpg', 'lora', 'fft', 'dept', 'prefix', 'ptuningv2'])

    ##################################################### Task #################################################### #
    parser.add_argument("--data_dir", type=str, default='./glue_data')
    parser.add_argument("--dataset_path", type=str, default='nyu-mll/glue', choices=['nyu-mll/glue', 'super_glue'])

    # #################################################### Wandb #################################################### #
    parser.add_argument('--wandb_logging', action='store_true', help="Log to wandb")
    parser.add_argument('--project_name', type=str, default='NLU')
    parser.add_argument('--run_name', type=str, default=None)

    # #################################################### PEFT #################################################### #
    parser.add_argument('--do_peft', type=int, default=None)

    # #################################################### Model #################################################### #
    parser.add_argument("--model_type", type=str, default=model_type)
    parser.add_argument("--enc_model_type", type=str, default=enc_model_type)

    # For LOPA
    parser.add_argument("--lp_rank", type=int, default=4, help="Rank of the decoded row/col vectors.")

    # For PHM - Currently only used by IDPG
    parser.add_argument("--use_phm_layers", type=bool, default=True)
    parser.add_argument("--phm_n", type=int, default=None, help="hyper-param n in kronecker product",
                        choices=[8, 16, 32])

    parser.add_argument("--use_fast_tokenizer", default=True,
                        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.", )
    parser.add_argument("--model_revision", type=str, default="main",
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--token", default=None,
                        help=(
                            "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                            "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."))
    parser.add_argument("--ignore_mismatched_sizes", default=False,
                        help="Whether or not to enable to load a pretrained model whose head dimensions are different.")

    # #################################################### Training ################################################# #
    # 1e-5 for LOPA and FFT (from RoBERTa paper),
    # 1e-4 for LoRA / PT / IDPG
    # 5e−5 for PrefixTuning
    # else choose from 5e−3,1e−3,5e−4,1e−4,5e−5,1e−5




    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--per_device_test_batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--dataset_name", type=str, default='trec',choices=['dbpedia', 'imdb', 'agnews','yelp','trec','yahoo'])

    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")



    #####  task prompt nums
    parser.add_argument("--num_virtual_tokens", default=4,type=int, )

    parser.add_argument("--instance_tokens", default=1,type=int, )


    parser.add_argument("--AL_method", type=str, default='hmask_cluster_calibration_entropy_local_div')

    parser.add_argument("--multi_head", type=int, default=4)


    ###### OOD相关
    parser.add_argument("--ood",action="store_true", help="是否使用ood模式"
    )

    parser.add_argument("--ood_name", type=str, default='sst2',choices=['sst2','imdb_contrast','imdb_counter'])




    ###### local div self.initial_alpha = 0.9 self.diversity_increase_factor = 0.1
    parser.add_argument("--initial_alpha", type=float, default=0.9)
    parser.add_argument("--diversity_increase_factor", type=float, default=0.1)
    ### 【按位积、self attention、按位加、only_task  only ins )
    parser.add_argument("--prompt_combine_method", type=str , default='self attention' )
    parser.add_argument("--vector", type=str , default='cls' ) #### hmask cls simcse












    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup",
                                 "inverse_sqrt", "reduce_lr_on_plateau"],
                        )
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Can also be defined within script!")
    parser.add_argument("--save_every", type=int, default=0)

    # ######################################## Testing (eval.py exclusive args) ##################################### #
    parser.add_argument("--eval_data_type", type=str, default='dev', choices=['dev', 'test'], help="Used in eval.py")
    parser.add_argument("--do_evaluation", type=bool, default=True,
                        help="Compute the evaluation metrics. Used in eval.py")
    parser.add_argument("--do_prediction", type=bool, default=True, help="Compute the predictions. Used in eval.py")
    parser.add_argument("--force_prediction_in_valid_labels", type=bool, default=True,
                        help="Force predictions in valid labels. Used in eval.py")

    # #################################################### Data #################################################### #

    parser.add_argument("--pad_to_max_length", default=False, help=" Otherwise, dynamic padding is used.")
    parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached features.")
    parser.add_argument("--dynamic_pad", type=bool, default=True,
                        help="Whether to use dynamic padding. DEPT cannot use dynamic padding.")

    # #################################################### Paths #################################################### #
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument('--load_base_from_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--load_adapter_from', type=str,
                        default='')
    parser.add_argument('--clf_predictor_path', type=str,
                        default='')

    # #################################################### Others ################################################ #
    parser.add_argument('--seed', type=int, default=619, help="random seed for initialization")
    parser.add_argument("--do_test", default=False, help="Whether to run testing.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str,
                        help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local mach"
        ),
    )

    # #################################################### Hardware ################################################ #
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--no_cuda", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training (multi-node): local_rank")
    parser.add_argument("--node_index", type=int, default=-1, help="node index if multi-node running")
    parser.add_argument("--gpu_per_node", type=int, default=4, help="num of gpus per node")

    # Data Loading when using datasets from the hub or datasets library [deprecated]
    parser.add_argument("--train_file", type=str, default=None,
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="A csv or a json file containing the validation data.")
    parser.add_argument("--train_split_name", type=str, default=None,
                        help='The name of the train split in the input dataset. '
                             'If not specified, will use the "train" split when do_train is enabled')
    parser.add_argument("--validation_split_name", type=str, default=None,
                        help='The name of the validation split in the input dataset. '
                             'If not specified, will use the "validation" split when do_eval is enabled')
    parser.add_argument("--test_split_name", type=str, default=None,
                        help='The name of the test split in the input dataset. '
                             'If not specified, will use the "test" split when do_predict is enabled')
    parser.add_argument("--remove_splits", type=str, default=None,
                        help="The splits to remove from the dataset. Multiple splits should be separated by commas.")
    parser.add_argument("--remove_columns", type=str, default=None,
                        help="The columns to remove from the dataset. Multiple columns should be separated by commas.")
    parser.add_argument("--label_column_name", type=str, default=None,
                        help="The name of the label column in the input dataset or a CSV/JSON file. "
                             "If not specified, will use the 'label' column for single/multi-label classification task")

    # parser.add_argument('--initial_se_factor',type=float,default=0.1 )

    args = parser.parse_args()
    # args.dataset_name = ALargs.dataset
    ALargs.label_ids = class_labels(args.dataset_name)
    ALargs.dataset = args.dataset_name
    ALargs.AL_method = args.AL_method
    ALargs.multi_head = args.multi_head
    ALargs.v_tokens_n = args.num_virtual_tokens
    ALargs.instance_tokens = args.instance_tokens
    ALargs.diversity_increase_factor =args.diversity_increase_factor
    ALargs.initial_alpha = args.initial_alpha
    ALargs.prompt_combine_method = args.prompt_combine_method
    ALargs.vector = args.vector


    # args.num_virtual_tokens = ALargs.v_tokens_n
    args.seed = random.randint(0, 1000)
    # Get huggingface paths for the models
    args.model_name_or_path = get_huggingface_path(args.model_type)
    args.config_name = get_huggingface_path(args.model_type)
    args.tokenizer_name = get_huggingface_path(args.model_type)
    args.enc_model_name_or_path = get_huggingface_path(args.enc_model_type)

    # # Get the task dataset
    # if args.dataset_name not in task_dir_mapping:
    #     raise ValueError(f"Data dir for the task {args.dataset_name} not specified.")
    # args.data_dir = os.path.join(args.data_dir, task_dir_mapping[args.dataset_name])

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    # Create a directory to store the logs
    if args.log_dir is None:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('./logging', str(args.dataset_name) + '-'+ ALargs.AL_method + '-' +str(ALargs.Q_size) +'-' + current_time + '-' + str(args.seed))
        args.log_dir = log_dir
    create_dir(args.log_dir)

    # Configure logging
    if is_rank_0():
        logging.basicConfig(filename=os.path.join(args.log_dir, 'logs.txt'), filemode='w',
                            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set save paths
    args.output_dir = os.path.join(args.log_dir, 'output')

    # Update the max_length and max_prompt_length by deducting the number of virtual tokens
    # args.max_length = args.max_length - args.num_virtual_tokens

    # Set the distributed training
    set_dist(args)
    # Set the seed
    ###AL 随机生成seed

    set_seed(args.seed)

    # Log the config
    config: dict = vars(args)
    config = {key: str(value) for key, value in config.items()}
    config = OrderedDict(sorted(config.items()))

    log_dist(message="\n\n# ############### PEFT ############## #\n\n", level=logging.INFO, ranks=[0])
    log_dist(message=json.dumps(config, indent=4), level=logging.INFO, ranks=[0])


    ######## AL args setting
    args.selected_index = []


    return args, logger
