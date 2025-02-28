

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

def get_product_scores(args,model_path):
    global_data = GlobalProdSearchData(args, args.data_dir, args.input_train_dir)
    test_prod_data = ProdSearchData(args, args.input_train_dir, "test", global_data)
    # model_path = os.path.join(args.save_dir, 'model_best.ckpt')
    best_model, _ = create_model(args, global_data, test_prod_data, model_path)
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_prod_data, args.rankfname)

if __name__ == "__main__":
    ## Parse command-line arguments
    model_to_test  = ".\trained_models\model_epoch_10.ckpt"
    args = parse_args()
    ## Read the config file
    config_args = Config()
    init_logger(os.path.join(config_args.save_dir, config_args.log_file))
    logger.info(config_args)
    get_product_scores(config_args,model_to_test)



