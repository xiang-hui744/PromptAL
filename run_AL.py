import json
from pprint import pprint

import torch
from accelerate.logging import MultiProcessAdapter
from tqdm import tqdm

from AL.best_acc_f1 import best_acc_f1
from AL.query_strategy import find_examples
from ALconfig import ALargs
from utils.config import get_config
from utils.custom import is_rank_0

import os

from utils.data_processors import task_dir_mapping

# Set the os environment
os.environ['PYTHONBREAKPOINT'] = '0'  # completely disable breakpoint()


def main():

    torch.cuda.set_device(0)
    args, logger = get_config()
    logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.

    # Get the task dataset
    if args.dataset_name not in task_dir_mapping:
        raise ValueError(f"Data dir for the task {args.dataset_name} not specified.")
    args.data_dir = os.path.join(args.data_dir, task_dir_mapping[args.dataset_name])


    # Print a big message for peft method
    if is_rank_0():
        print("\n\n")
        print("#" * 100)
        print(f"PEFT Method: {args.peft_method}")
        print("#" * 100)


    ### 记录每次主动学习迭代选择的样本编号
    index_dict = {}



    ############## 只有样本提示
    # ALargs.prompt_combine_method = 'only ins'







    ##########  开始主动学习迭代  ##########
    for iteration in tqdm(range(ALargs.iterations),colour='yellow'):

        ALargs.current_iterations = iteration
        if args.peft_method == 'lopa':
            from trainers.dspal import Trainer

            ##### 每次主动学习都是初始化全新的模型进行训练
            trainer = Trainer(args, logger)
            ##### AL
            # ALargs.total_trainer = trainer

        else:
            raise ValueError(f"PEFT method {args.peft_method} currently not supported.")
        print(f'\n\n!!!! 开始主动学习第 {iteration} 次迭代')
        logger.info(f'\n\n!!!! 开始主动学习第 {iteration} 次迭代')
        logger.info("\n######   args:")
        logger.info(args)
        print("\n######   args:")
        print(args)
        logger.info("\n######   ALargs:")
        logger.info(ALargs)
        print("\n######   ALargs:")
        print(ALargs)


        ########  选择本次主动学习迭代的样本索引 k_index
        ALargs.pooling = True
        ALargs.h_mask = torch.empty(0, 768,device=torch.device('cuda'))


        ####### 第0次主动学习迭代 应该是随机选择样本
        if ALargs.current_iterations == 0:
            ALargs.total_trainer = None
            # ALargs.total_trainer = trainer

        ###### 使用上一次的模型进行样本选择 初试第一次主动学习则随机选择训练集
        k_index = find_examples()


        print("\n!!!! h_mask size:",ALargs.h_mask.shape)

        ALargs.pooling = False

        ### 记录选择的样本
        index_dict[iteration] = k_index
        print(f'!!!! 第{iteration}迭代 AL迭代选择了 {len(k_index)} 个样本,如下')
        logger.info(f'!!!!  第{iteration}迭代迭代选择了 {len(k_index)} 个样本,如下')
        print(k_index)
        logger.info(k_index)

        ########  本次主动学习选择的样本和之前轮次的主动学习样本一起形成本次模型的训练集
        ALargs.selected_index.extend(k_index)

        print(f'\n!!!!  AL总共选择了 {len(ALargs.selected_index)} 个样本')
        logger.info(f'\n!!!!  AL总共选择了 {len(ALargs.selected_index)} 个样本')
        print(f'!!!! 第{iteration}迭代 AL总共选择了 {len(k_index)} 个样本,如下')
        logger.info(f'!!!!  第{iteration}迭代 AL总共选择了 {len(k_index)} 个样本,如下')
        print(ALargs.selected_index)
        logger.info(ALargs.selected_index)
        ### 形成train set和 train dataloader

        print(f'\n!!!! 形成train set和 train dataloader')
        logger.info(f'\n!!!! 形成train set和 train dataloader')
        trainer.get_train_dataset()
        trainer.build_dataloader(task='train')

        ####### 开始本次主动学习迭代以及测试
        print(f'\n!!!! 开始本次主动学习迭代以及测试')
        logger.info(f'\n!!!! 开始本次主动学习迭代以及测试')
        trainer.get_optimizer()
        trainer.train_loop()
        ALargs.total_trainer = trainer





    index_json_path= os.path.join(args.log_dir, 'select_index.json')
    test_json_path = os.path.join(args.log_dir, 'test_result.json')
    with open(index_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_dict,f,ensure_ascii=False,indent=4)

    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(ALargs.test_results, f, ensure_ascii=False, indent=4)


    ##### result logging
    total_path = f'./best_result'

    args_path  = ALargs.get_path()
    acc_dict, f1_dict = best_acc_f1(ALargs.test_results,seed = args.seed)
    os.makedirs(total_path, exist_ok=True)
    with open(total_path+'acc_'+args_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(acc_dict,ensure_ascii=False)+"\n")
    with open(total_path+'f1_'+args_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(f1_dict,ensure_ascii=False)+"\n")


if __name__ == '__main__':

    main()
