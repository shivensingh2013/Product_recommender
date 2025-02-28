## Executes the training pipeline.
import torch
import argparse
import random
import glob
import os
import yaml

from others.logging import logger, init_logger
from models.ps_model import ProductRanker, build_optim
from models.item_transformer import ItemTransformerRanker
from data.data_util import GlobalProdSearchData, ProdSearchData
from trainer import Trainer
from data.prod_search_dataset import ProdSearchDataset
from main import create_model
from config import Config
from utils import parse_args,load_config


def train(args):
    args.start_epoch = 0
    logger.info('Device %s' % args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    global_data = GlobalProdSearchData(args, args.data_dir, args.input_train_dir)
    train_prod_data = ProdSearchData(args, args.input_train_dir, "train", global_data)
    #subsampling has been done in train_prod_data
    model, optim = create_model(args, global_data, train_prod_data, args.train_from)
    trainer = Trainer(args, model, optim)
    valid_prod_data = ProdSearchData(args, args.input_train_dir, "valid", global_data)
    best_checkpoint_path = trainer.train(trainer.args, global_data, train_prod_data, valid_prod_data)
    test_prod_data = ProdSearchData(args, args.input_train_dir, "test", global_data)
    best_model, _ = create_model(args, global_data, train_prod_data, best_checkpoint_path)
    del trainer
    torch.cuda.empty_cache()
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_prod_data, args.rankfname)

if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_args()
    
    ## Read the config file
    config_args = Config()
    ## Save the directories
    if not os.path.isdir(config_args.save_dir):
        os.makedirs(config_args.save_dir)
    init_logger(os.path.join(config_args.save_dir, config_args.log_file))
    logger.info(config_args)
    train(config_args)

    