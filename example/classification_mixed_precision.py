"""
@Time : 2022/8/17 23:18 
@Author : sunshb10145 
@File : classification_mixed_precision.py 
@desc:
"""
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""
一行代码开启混合精度，加速训练
tips: 1.只支持tf.train.Optimizer or tf.keras.optimizers.Optimizer继承来的，不支持keras 原生的optimizer
      2. 修改opt 放在build model 前，否则某些情况会报错
训练速度能提高约30% 左右，244ms/step -> 168ms/step
"""
import os

os.environ['TF_KERAS'] = '1'  # 使用tf.keras

from tqdm import tqdm
import numpy as np

np.random.seed(42)
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import *
from keras.models import Model
from config import *
from data_utils import load_data, data_generator, Evaluator

# 加载数据集
train_data = load_data(
    data_dict.get(task_name + "_train"), task_name
)
valid_data = load_data(
    data_dict.get(task_name + "_val"), task_name
)
# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

train_generator = data_generator(train_data, task_name, tokenizer, batch_size=batch_size)
val_generator = data_generator(valid_data, task_name, tokenizer, batch_size=batch_size)

# create opt before build model
opt = Adam(lr)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)  # 开启混合精度

# build model
bert = build_transformer_model(config_path=config_path,
                               checkpoint_path=checkpoint_path,
                               num_hidden_layers=num_hidden_layers)
output = Lambda(lambda x: x[:, 0])(bert.output)
output = Dense(num_classes, activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['sparse_categorical_accuracy'])

if __name__ == '__main__':
    # evaluator = Evaluator(val_generator)
    callbacks = [
        Evaluator(val_generator),
        EarlyStopping(
            monitor='val_acc',
            patience=5,
            verbose=1,
            mode='max'),
        ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-5,
            mode='max'),
        # ModelCheckpoint(
        #     f'best_model.h5',
        #     monitor='val_acc',
        #     save_weights_only=True,
        #     save_best_only=True,
        #     verbose=1,
        #     mode='max'),
    ]
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # callbacks=[evaluator],
        callbacks=callbacks
    )
else:
    model.load_weights('best_baseline.weights')
