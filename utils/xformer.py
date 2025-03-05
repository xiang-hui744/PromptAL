import logging

import numpy as np
from transformers import RobertaConfig, T5Config, BartConfig, GPT2Config, OpenAIGPTConfig, BertConfig, \
	DistilBertConfig, GPTNeoConfig, AutoConfig, AutoModelForSequenceClassification, BertModel
from transformers import RobertaModel, T5ForConditionalGeneration, BartForConditionalGeneration, GPT2LMHeadModel, \
	OpenAIGPTLMHeadModel, BertForMaskedLM, DistilBertForMaskedLM, GPTNeoForCausalLM
from transformers import RobertaTokenizer, T5Tokenizer, BartTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer, \
	BertTokenizer, DistilBertTokenizer, AutoTokenizer

from utils.custom import is_rank_0

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
	'roberta-base': (RobertaConfig, RobertaModel),
	'roberta-large': (RobertaConfig, RobertaModel),
	'bert-base-uncased':(BertConfig, BertModel),
	't5': (T5Config, T5ForConditionalGeneration),
	'bart': (BartConfig, BartForConditionalGeneration),
	'gpt2': (GPT2Config, GPT2LMHeadModel),
	'gpt2-medium': (GPT2Config, GPT2LMHeadModel),
	'gpt2-large': (GPT2Config, GPT2LMHeadModel),
	'gpt2-xl': (GPT2Config, GPT2LMHeadModel),
	'gpt-neo-125M': (GPTNeoConfig, GPTNeoForCausalLM),
	'gpt-neo-1.3B': (GPTNeoConfig, GPTNeoForCausalLM),
	'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
	'bert': (BertConfig, BertForMaskedLM),
	'distilbert': (DistilBertConfig, DistilBertForMaskedLM),
}

TOKENIZER_CLASSES = {
	'roberta-base': RobertaTokenizer,
	'roberta-large': RobertaTokenizer,
	'bert-base-uncased':BertTokenizer,
	't5': T5Tokenizer,
	'bart': BartTokenizer,
	'gpt2': GPT2Tokenizer,
	'gpt2-medium': GPT2Tokenizer,
	'gpt2-large': GPT2Tokenizer,
	'gpt2-xl': GPT2Tokenizer,
	'gpt-neo-125M': GPT2Tokenizer,
	'gpt-neo-1.3B': GPT2Tokenizer,
	'openai-gpt': OpenAIGPTTokenizer,
	'bert': BertTokenizer,
	'distilbert': DistilBertTokenizer,
}


LORA_IA3_TARGET_MODULES = {
	# ############################# Microsoft Phi Models ############################## #
	"roberta-large": {
		"target_modules_lora": ["query", "key", "value"],
	}
}


def get_model_size(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	model_size = sum([np.prod(p.size()) for p in model_parameters])
	return "{}M".format(round(model_size / 1e+6))


def load_tokenizer(args, model_type, tokenizer_name):
	if model_type in TOKENIZER_CLASSES:
		tokenizer_class = TOKENIZER_CLASSES[model_type]
	else:
		tokenizer_class = AutoTokenizer
		if is_rank_0():
			print("Using AutoTokenizer for model_type: ", model_type)
	
	tokenizer = tokenizer_class.from_pretrained(
		tokenizer_name,
		use_fast=args.use_fast_tokenizer,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
		revision=args.model_revision,
		token=args.token,
	)
	
	# Some Tokenizers do not have pad_token. We add it here. (It will only be used for ease of use in my pipeline.)
	if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
		tokenizer.pad_token = tokenizer.eos_token
	
	if is_rank_0():
		logger.info("Finish loading Tokenizer from %s", tokenizer_name)
	return tokenizer


def load_base_model(args, model_type, model_name_or_path, config_class=None, model_class=None,only_config=False):
	
	if config_class is None or model_class is None:
		config_class, model_class = MODEL_CLASSES[model_type]
	
	# With the config we initialize the model to do classification for config.num_labels (used by model's RobertaClassificationHead)
	config = config_class.from_pretrained(
		model_name_or_path,  # [Note] Not using config_name since it is the same as model_name_or_path
		# num_labels=args.num_labels,  # [Note] Unless we load a ModelForSequenceClassification, this is not used
		finetuning_task="text-classification",
		cache_dir=args.cache_dir,
		revision=args.model_revision,
		token=args.token,
		trust_remote_code=args.trust_remote_code
	)
	
	if args.is_regression:
		config.problem_type = "regression"
		logger.info("setting problem type to regression")
	elif args.is_multi_label:
		config.problem_type = "multi_label_classification"
		logger.info("setting problem type to multi label classification")
	else:
		config.problem_type = "single_label_classification"
		logger.info("setting problem type to single label classification")
	# model:RobertaForMaskedLM  config:RobertaConfig

	if only_config:
		return config,None
	model = model_class.from_pretrained(
		model_name_or_path,
		from_tf=bool(".ckpt" in model_name_or_path),
		config=config,
		cache_dir=args.cache_dir,
		revision=args.model_revision,
		token=args.token,
		trust_remote_code=args.trust_remote_code,
		ignore_mismatched_sizes=args.ignore_mismatched_sizes,
	)
	
	if is_rank_0():
		logger.info("Finish loading Base model [%s] from %s", get_model_size(model), model_name_or_path)
	return config, model


def get_huggingface_path(model: str) -> str:
	# ############################# FacebookAI BERT Models ############################# #
	if model == 'roberta-large':
		# huggingface_path = 'roberta-large'  # roberta-large (355M)
		huggingface_path = '/mnt/10TB/models/roberta-large'
	elif model == 'roberta-base':
		# huggingface_path = 'roberta-base'  # roberta-base (125M)
		huggingface_path = '/mnt/10TB/models/roberta-base'

	elif model == "bert-base-uncased":
		huggingface_path = '/mnt/10TB/models/bert-base-uncased'

	# ############################# OpenAI GPT Models ############################# #
	elif model == 'gpt2':  # gpt2 (124M)
		# huggingface_path = 'gpt2'
		huggingface_path = '/mnt/10TB/models/gpt2/'

	elif model == 'gpt2-medium':  # gpt2-medium (345M)
		huggingface_path = 'gpt2-medium'
	elif model == 'gpt2-large':  # gpt2-large (774M)
		huggingface_path = 'gpt2-large'
	elif model == 'gpt2-xl':   # gpt2-xl (1.5B)
		huggingface_path = 'gpt2-xl'
	elif model == 'gpt-neo-125M':
		huggingface_path = 'EleutherAI/gpt-neo-125M'
	elif model == 'gpt-neo-1.3B':
		huggingface_path = 'EleutherAI/gpt-neo-1.3B'  # 'EleutherAI/gpt-neo-1.3B' or 'EleutherAI/gpt-neo-2.7B'
	else:
		raise NotImplementedError()
	
	return huggingface_path
