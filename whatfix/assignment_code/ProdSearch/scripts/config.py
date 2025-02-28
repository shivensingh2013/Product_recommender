class Config:
    def __init__(self):
        # Hyperparameters and configurations
        self.seed = 323
        self.train_from = ""
        self.model_name = "item_transformer"
        # Choices - ['review_transformer', 'item_transformer', 'ZAM', 'AEM', 'QEM']
        self.sep_prod_emb = False
        self.pretrain_emb_dir = ""
        self.pretrain_up_emb_dir = ""
        self.fix_train_review = False
        self.do_seq_review_train = False
        self.do_seq_review_test = False
        self.fix_emb = False
        self.use_dot_prod = True
        self.sim_func = "product"
        self.use_pos_emb = True
        self.use_seg_emb = True
        self.use_item_pos = False
        self.use_item_emb = False
        self.use_user_emb = False
        self.rankfname = "test.best_model.ranklist"
        self.dropout = 0.1
        self.token_dropout = 0.1
        self.optim = "adam"
        self.lr = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.decay_method = "adam"
        self.warmup_steps = 8000
        self.max_grad_norm = 5.0
        self.subsampling_rate = 1e-5
        self.do_subsample_mask = False
        self.prod_freq_neg_sample = False
        self.pos_weight = False
        self.l2_lambda = 0.0
        self.batch_size = 64
        self.has_valid = False
        self.valid_batch_size = 24
        self.valid_candi_size = 500
        self.test_candi_size = -1
        self.candi_batch_size = 500
        self.num_workers = 4
        self.data_dir = './amazon_data_processed/cell_phone/temp_data/min_count5/'
        self.input_train_dir = './amazon_data_processed/cell_phone/temp_data/min_count5/seq_query_split/'
        self.save_dir = './model_checkpoints/'
        self.log_file = "train.log"
        self.query_encoder_name = "fs"
        self.review_encoder_name = "pvc"
        self.embedding_size = 128
        self.ff_size = 512
        self.heads = 8
        self.inter_layers = 2
        self.review_word_limit = 100
        self.uprev_review_limit = 20
        self.iprev_review_limit = 30
        self.pv_window_size = 1
        self.corrupt_rate = 0.9
        self.shuffle_review_words = True
        self.train_review_only = True
        self.max_train_epoch = 5
        self.train_pv_epoch = 0
        self.start_epoch = 0
        self.steps_per_checkpoint = 200
        self.neg_per_pos = 5
        self.sparse_emb = False
        self.scale_grad = False
        self.weight_distort = False
        self.mode = "train"
        self.rank_cutoff = 100
        self.device = "cuda"

    # Optionally, you can add a method to print the configuration or access attributes easily
    def print_config(self):
        for attribute, value in self.__dict__.items():
            print(f'{attribute}: {value}')
