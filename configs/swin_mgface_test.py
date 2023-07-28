from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)  # 损失的超参数
config.network = "swin_mgface"
config.resume = False
config.embedding_size = 512
config.sample_rate = 1.0  # partial_fc的超参数
config.fp16 = True
config.momentum = 0.9
# config.weight_decay = 0.05
config.weight_decay = 5e-4
config.batch_size = 32
config.optimizer = "adamw"
config.lr = 1.25e-4
config.verbose = 2000  # 多少个step测试一次

config.rec = "../../raw_data"  # 训练测试数据的根目录
config.num_epoch = 20
config.warmup_epoch = 0
config.train_targets = ['100']
config.val_targets = ['lfw_align', 'mask_lfw_align']
config.max_classes = 8000  # 最多使用多少人训练，-1表示使用所有人
config.output = "swin_mgface"
# config.teacher_output = "ms1mv3_arcface_swin_b"
config.loss_list = (1.0, 0.01, 0.1)
