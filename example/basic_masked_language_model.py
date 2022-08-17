#! -*- coding: utf-8 -*-

"""
@Time : 2022/8/17 14:48 
@Author : sunshb10145 
@File : basic_masked_language_model.py 
@desc: 测试代码可用性: MLM
"""
import shutil
from bert4keras.models import build_transformer_model
from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer

config_path = '../saved_model/bert_config.json'
shutil.copy("../model/bert_config.json", config_path)
checkpoint_path = './saved_model/bert_model.ckpt'
dict_path = '../saved_model/vocab.txt'
shutil.copy("../model/vocab.txt", dict_path)

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)  # 建立模型，加载权重

token_ids, segment_ids = tokenizer.encode(u'北京四维图新科技有限公司')

# mask掉“技术”
token_ids[9] = token_ids[10] = tokenizer._token_mask_id
token_ids, segment_ids = to_array([token_ids], [segment_ids])

# 用mlm模型预测被mask掉的部分
probas = model.predict([token_ids, segment_ids])[0]
print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“有限”
