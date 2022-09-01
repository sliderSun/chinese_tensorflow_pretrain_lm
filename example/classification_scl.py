"""
@Time : 2022/8/17 23:09 
@Author : sunshb10145 
@File : classification_scl.py 
@desc:
"""

"""
ref: [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](http://arxiv.org/abs/2011.01403)
"""

import keras
from bert4keras.backend import K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np

np.random.seed(42)

from config import *
from data_utils import load_data

# 加载数据集
train_data = load_data(
    data_dict.get(task_name + "_train"), task_name
)
valid_data = load_data(
    data_dict.get(task_name + "_val"), task_name
)

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class SubDataGenerator(DataGenerator):
    def take(self, nums=1, shuffle=False):
        """take nums * batch examples"""
        d = []
        for i, data in enumerate(self.__iter__(shuffle)):
            if i >= nums:
                break

            d.append(data)

        if nums == 1:
            return d[0]
        return d


class data_generator(SubDataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_desc) in self.sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([int(label)])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels], None

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(train_data, batch_size=batch_size)
val_generator = data_generator(valid_data, batch_size=batch_size)


def evaluate(data):
    total, right = 0., 0.
    for d, _ in tqdm(data):
        x_true, y_true = d[:2], d[-1]
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


class SupervisedContrastiveLearning(Loss):
    """https://arxiv.org/pdf/2011.01403.pdf"""

    def __init__(self, alpha=1., T=1., **kwargs):
        super(SupervisedContrastiveLearning, self).__init__(**kwargs)
        self.alpha = alpha  # loss weight
        self.T = T  # Temperature

    def compute_loss(self, inputs, mask=None):
        loss = self.compute_loss_of_scl(inputs)
        loss = loss * self.alpha
        self.add_metric(loss, name='scl_loss')
        return loss

    def get_label_mask(self, y_true):
        """获取batch内相同label样本"""
        label = K.cast(y_true, 'int32')
        label_2 = K.reshape(label, (1, -1))
        mask = K.equal(label_2, label)
        mask = K.cast(mask, K.floatx())
        mask = mask * (1 - K.eye(K.shape(y_true)[0]))  # 排除对角线，即 i == j
        return mask

    def compute_loss_of_scl(self, inputs, mask=None):
        y_pred, y_true = inputs
        label_mask = self.get_label_mask(y_true)
        y_pred = K.l2_normalize(y_pred, axis=1)  # 特征向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线,即 i == j

        similarities = similarities / self.T  # Temperature scale
        similarities = K.exp(similarities)  # exp

        sum_similarities = K.sum(similarities, axis=-1, keepdims=True)  # sum i != k
        scl = similarities / sum_similarities
        scl = K.log(scl + K.epsilon())  # sum log
        scl = -K.sum(scl * label_mask, axis=1, keepdims=True) / (K.sum(label_mask, axis=1, keepdims=True) + K.epsilon())
        return K.mean(scl)


class CrossEntropy(Loss):
    def __init__(self, alpha, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, inputs, mask=None):
        pred, ytrue = inputs
        acc = keras.metrics.sparse_categorical_accuracy(ytrue, pred)
        self.add_metric(acc, name='clf_acc')

        ytrue = K.cast(ytrue, 'int32')
        ytrue = K.one_hot(ytrue, num_classes=num_classes)
        ytrue = K.reshape(ytrue, (-1, num_classes))
        loss = ytrue * K.log(pred + K.epsilon()) + (1 - ytrue) * K.log(1 - pred + K.epsilon())
        loss = -K.mean(loss)
        loss = loss * self.alpha
        self.add_metric(loss, name='clf_loss')

        return loss


# build model
bert = build_transformer_model(config_path=config_path,
                               checkpoint_path=checkpoint_path,
                               num_hidden_layers=num_hidden_layers,
                               return_keras_model=False,
                               )
output = Lambda(lambda x: x[:, 0])(bert.output)

y_in = Input(shape=(None,))

# scale_output = Dense(256, kernel_initializer=bert.initializer)(output)
# logits = Dense(num_classes)(output)
scl_output = SupervisedContrastiveLearning(alpha=0.05, T=0.05, output_axis=[0])([output, y_in])

clf_output = Dense(num_classes, activation='softmax')(output)
clf_ce = CrossEntropy(output_axis=[0], alpha=0.95)([clf_output, y_in])
model = Model(bert.inputs, clf_output)
model.summary()

train_model = Model(bert.inputs + [y_in], [scl_output, clf_ce])
train_model.compile(
    optimizer=Adam(lr)
)

if __name__ == '__main__':
    evaluator = Evaluator()
    train_model.fit_generator(train_generator.forfit(),
                              steps_per_epoch=len(train_generator),
                              epochs=epochs,
                              callbacks=[evaluator])

    # tsne
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    f = K.function(bert.inputs, output)

    d, _ = val_generator.take()
    x, y = d[:2], d[-1]

    logits = f(x)
    tsne = TSNE(n_components=2, learning_rate=100)
    tsne.fit_transform(logits)

    y = y.reshape((-1,)).tolist()
    plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=y)
