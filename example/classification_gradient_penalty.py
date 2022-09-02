#! -*- coding:utf-8 -*-
# 通过梯度惩罚增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234
# 适用于Keras 2.3.1

import json
from bert4keras.backend import keras, search_layer, K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Lambda, Dense
from tqdm import tqdm
import numpy as np

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

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 转换数据集
train_generator = data_generator(train_data, task_name, tokenizer, batch_size=batch_size)
val_generator = data_generator(valid_data, task_name, tokenizer, batch_size=batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(lr),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


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
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # callbacks=[evaluator],
        callbacks=callbacks
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
