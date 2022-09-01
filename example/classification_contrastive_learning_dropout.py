"""
@Time : 2022/8/17 23:22 
@Author : sunshb10145 
@File : classification_contrastive_learning_dropout.py 
@desc:
"""
"""
用dropout 做数据增强，构造样本的不同view，来通过增加对比学习增强分类模型的性能
"""
from bert4keras.snippets import DataGenerator, sequence_padding
import numpy as np
from tqdm import tqdm

np.random.seed(42)
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import *
from bert4keras.layers import *
from keras.losses import kullback_leibler_divergence as kld
from keras.models import Model
from config import *
from data_utils import load_data

# 加载数据集
train_data = load_data(
    data_dict.get(task_name + "_train"), task_name
)
valid_data = load_data(
    data_dict.get(task_name + "_val"), task_name
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(data=train_data, batch_size=batch_size)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)


class TotalLoss(Loss):
    "计算两部分loss：分类交叉熵和对比loss"

    def compute_loss(self, inputs, mask=None):
        cls_loss = self.sparse_categorical_crossentropy(inputs)
        sim_loss = self.simcse_loss(inputs)
        return cls_loss + sim_loss / 4 * 4

    def simcse_loss(self, inputs):
        """用于SimCSE训练的loss
        """
        # 构造标签
        _, _, _, _, y_pred = inputs
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = K.equal(idxs_1, idxs_2)
        y_true = K.cast(y_true, K.floatx())
        # 计算相似度
        y_pred = K.l2_normalize(y_pred, axis=1)
        similarities = K.dot(y_pred, K.transpose(y_pred))
        similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
        loss = K.mean(loss)
        self.add_metric(loss, 'sim_loss')
        return loss

    def sparse_categorical_crossentropy(self, inputs, mask=None):
        _, _, y_true, _, y_pred = inputs
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        loss = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
        self.add_metric(loss, 'cls_loss')
        return loss

    def crossentropy_with_rdrop(self, inputs, alpha=4):
        """配合R-Drop的交叉熵损失
        """
        _, _, y_true, _, y_pred = inputs
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
        loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        loss = loss1 + K.mean(loss2) / 4 * alpha
        self.add_metric(loss, 'rdrop')
        return loss

    def compute_kld(self, inputs, alpha=4, mask=None):
        _, _, _, y_pred = inputs
        loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        loss = K.mean(loss) / 4 * alpha
        self.add_metric(loss, 'kld')
        return loss


bert = build_transformer_model(checkpoint_path=checkpoint_path,
                               config_path=config_path,
                               dropout_rate=0.3,
                               return_keras_model=False
                               )

label_inputs = Input(shape=(None,), name='label_inputs')

pooler = Lambda(lambda x: x[:, 0])(bert.output)
x = Dense(units=num_classes, activation='softmax', name='classifier')(pooler)
output = TotalLoss(4)(bert.inputs + [label_inputs, pooler, x])

model = Model(bert.inputs + [label_inputs], output)
classifier = Model(bert.inputs, x)

model.compile(optimizer=Adam(lr), metrics=['sparse_categorical_accuracy'])
model.summary()


def evaluate(val_data=valid_generator):
    total = 0.
    right = 0.
    for (x, s, y_true), _ in tqdm(val_data):
        y_pred = classifier.predict([x, s]).argmax(axis=-1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    print(total, right)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path='best_model.weights'):
        self.best_val_acc = 0.
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.save_path)

        print('current acc :{}, best val acc: {}'.format(val_acc, self.best_val_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_generator.forfit(),
              steps_per_epoch=len(train_generator),
              epochs=epochs,
              callbacks=[evaluator])
