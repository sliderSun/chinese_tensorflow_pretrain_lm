"""
@Time : 2022/9/1 10:20 
@Author : sunshb10145 
@File : config.py 
@desc:
"""

num_classes = 16  # 119 iflytek
maxlen = 64
batch_size = 32
epochs = 20
num_hidden_layers = 12
lr = 2e-5
pretrain_epochs = 10
fine_tune_epochs = 5
pretrain_lr = 5e-5
fine_tune_lr = 2e-5
task_name = "tnews"  # "iflytek"

# BERT base
base_dir = "D:\sunshubing\sw\chinese_tensorflow_pretrain_lm\model"
config_path = base_dir + "\\bert_config.json"
checkpoint_path = base_dir + "\\bert_model.ckpt"
dict_path = base_dir + "\\vocab.txt"
