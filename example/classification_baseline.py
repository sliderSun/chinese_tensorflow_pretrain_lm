"""
@Time : 2022/8/29 22:27 
@Author : sunshb10145 
@File : classification_baseline.py.py 
@desc:
tnews baseline:
bert-12 acc: 56.6
tnews: (https://github.com/CLUEbenchmark/CLUE)
"""
from bert4keras.layers import *
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer
from keras.models import Model
from tqdm import tqdm
import numpy as np

np.random.seed(42)
from config import *
from data_utils import load_data, data_generator

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


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()

    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(val_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights('best_baseline.weights')
        print('acc: {}, best acc: {}'.format(acc, self.best_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])
else:
    model.load_weights('best_baseline.weights')
