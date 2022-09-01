"""
@Time : 2022/9/1 10:20 
@Author : sunshb10145 
@File : config.py 
@desc:
"""
n = 5
SEED = 42

num_classes = 16  # 119 iflytek
maxlen = 64
batch_size = 32
epochs = 20
num_hidden_layers = 12

max_segment = 2
grad_accum_steps = 64
drop = 0.2

lr = 2e-5

predecessor_epochs = 5
successor_epochs = 5
theseus_epochs = 10
pretrain_epochs = 10
fine_tune_epochs = 20  # 5

pretrain_lr = 5e-5
fine_tune_lr = 2e-5
task_name = "tnews"  # "iflytek"
data_dict = {
    "tnews_train": "./tnews_public/train.json",
    "tnews_val": "./tnews_public/dev.json",
    "iflytek_train": "./iflytek_public/train.json",
    "iflytek_val": "./iflytek_public/dev.json"
}
# BERT base
base_dir = "../model"
config_path = base_dir + "\\bert_config.json"
checkpoint_path = base_dir + "\\bert_model.ckpt"
dict_path = base_dir + "\\vocab.txt"
