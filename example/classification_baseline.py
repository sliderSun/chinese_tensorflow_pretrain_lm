"""
@Time : 2022/8/29 22:27 
@Author : sunshb10145 
@File : classification_baseline.py.py 
@desc:
tnews baseline:
bert-12 acc: 56.6
tnews: (https://github.com/CLUEbenchmark/CLUE)
"""
import numpy as np
from bert4keras.layers import *
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

np.random.seed(42)
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

# build model
bert = build_transformer_model(config_path=config_path,
                               checkpoint_path=checkpoint_path,
                               num_hidden_layers=num_hidden_layers)
output = Lambda(lambda x: x[:, 0])(bert.output)
output = Dense(num_classes, activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr),
              metrics=['sparse_categorical_accuracy'])

if __name__ == '__main__':
    # evaluator = Evaluator()
    callbacks = [
        Evaluator(),
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
