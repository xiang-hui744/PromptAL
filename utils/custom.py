import os
import pathlib
import logging
from typing import List
import torch
import multiprocessing

logger = logging.getLogger(__name__)


def is_rank_0() -> bool:
	# Can also use accelerator.is_local_main_process if using Accelerator
	return int(os.environ.get("RANK", "0")) == 0


def create_dir(path: str):
	if not is_rank_0():
		return
	_dir = pathlib.Path(path)
	_dir.mkdir(parents=True, exist_ok=True)
	
	
def log_dist(
		message: str,
		ranks: List[int],
		level: int = logging.INFO
) -> None:
	"""Log messages for specified ranks only"""
	my_rank = int(os.environ.get("RANK", "0"))
	if my_rank in ranks:
		if level == logging.INFO:
			logger.info(f'[Rank {my_rank}] {message}')
		if level == logging.ERROR:
			logger.error(f'[Rank {my_rank}] {message}')
		if level == logging.DEBUG:
			logger.debug(f'[Rank {my_rank}] {message}')


def set_dist(args):
	# To train on cpu, set args.no_cuda=True else it will use all available gpus [Recommended use for now]
	if args.local_rank == -1 or args.no_cuda:
		if torch.cuda.is_available():
			device = torch.device("cuda")
			args.n_gpu = torch.cuda.device_count()
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			args.n_gpu = 1
		else:
			device = torch.device("cpu")
			args.n_gpu = 0
	# To enable distributed training (does it mean multi-node?), set local_rank
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		# noinspection PyUnresolvedReferences
		torch.distributed.init_process_group(backend='nccl')
		args.local_rank += args.node_index * args.gpu_per_node
		args.n_gpu = 1
	
	cpu_cont = multiprocessing.cpu_count()  # Gives number of logical CPU cores
	# Do not use all cpu cores for parallel processing. For computationally intensive tasks, recommended usage is
	# to use number of physical CPU cores i.e. = (number of logical CPU cores)/2
	# Recommended reading: https://superfastpython.com/multiprocessing-pool-num-workers/
	args.cpu_cont = cpu_cont - int(cpu_cont / 2)  # Ignore half of the cores
	args.device = device
	
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count(using): %d, "
				   "cpu count(available): %d", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1),
				   args.cpu_cont, cpu_cont)