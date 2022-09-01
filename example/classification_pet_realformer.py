"""
@Time : 2022/8/29 22:50 
@Author : sunshb10145 
@File : classification_pet_realformer.py 
@desc:
"""
"""
残差式attention RealFormer实验，由于结构与BERT上有差异，直接加载BERT权重后fine-tuning对比是不公平的，不过为了简单，
所以选了pet这种“预训练”+“fine-tuning" 一体的模型结构实验，此外，为了让模型有更多的机会调整权重，epoch设置的大一些
best acc: 56.73
"""

from bert4keras.layers import *
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import *
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.tokenizers import Tokenizer
from keras.models import Model
from tqdm import tqdm
import numpy as np

np.random.seed(42)
from example.config import *
from example.data_utils import load_data

desc2label = {
    'news_agriculture': '农业',
    'news_car': '汽车',
    'news_culture': '文化',
    'news_edu': '教育',
    'news_entertainment': '娱乐',
    'news_finance': '财经',
    'news_game': '游戏',
    'news_house': '房产',
    'news_military': '军事',
    'news_sports': '体育',
    'news_stock': '股市',
    'news_story': '民生',
    'news_tech': '科技',
    'news_travel': '旅游',
    'news_world': '国际'
}
labels = list(desc2label.values())

# 加载数据集
train_data = load_data(
    './tnews_public/train.json', task_name
)
valid_data = load_data(
    './tnews_public/dev.json', task_name
)

tokenizer = Tokenizer(dict_path, do_lower_case=True)

# pattern
pattern = '下面是一条体育新闻'
mask_idx = [6, 7]


class data_generator(DataGenerator):

    def random_masking(self, tokens):
        rands = np.random.random(len(tokens))
        source, target = [], []
        for r, t in zip(rands, tokens):
            if r < 0.15 * 0.8:
                source.append(tokenizer._token_mask_id)
                target.append(t)
            elif r < 0.15 * 0.9:
                source.append(t)
                target.append(t)
            elif r < 0.15:
                source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(t)

        return source, target

    def __iter__(self, shuffle=False):
        batch_tokens, batch_segments, batch_targets = [], [], []

        for is_end, (text, _, label) in self.sample(shuffle):
            text = pattern + text
            token_ids, seg_ids = tokenizer.encode(text, maxlen=maxlen)
            # 训练时随机masking
            if shuffle:
                source_tokens, target_tokens = self.random_masking(token_ids)
            else:
                source_tokens, target_tokens = token_ids[:], token_ids[:]

            # mask label
            if len(label) == 2:
                label_ids = tokenizer.encode(label)[0][1:-1]
                for m, l in zip(mask_idx, label_ids):
                    source_tokens[m] = tokenizer._token_mask_id
                    target_tokens[m] = l

            batch_tokens.append(source_tokens)
            batch_segments.append(seg_ids)
            batch_targets.append(target_tokens)

            if len(batch_tokens) == self.batch_size or is_end:
                batch_tokens = sequence_padding(batch_tokens)
                batch_segments = sequence_padding(batch_segments)
                batch_targets = sequence_padding(batch_targets)

                yield [batch_tokens, batch_segments, batch_targets], None

                batch_tokens, batch_segments, batch_targets = [], [], []


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)

        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * y_mask) / K.sum(y_mask)
        self.add_metric(acc, name='sparse_categorical_accuracy')
        return loss


model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                with_mlm=True,
                                with_residual_attention=True)

target_in = Input(shape=(None,))
output = CrossEntropy(1)([target_in, model.output])

train_model = Model(model.inputs + [target_in], output)
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()

label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])


def evaluate(data):
    total, right = 0., 0.
    for x, _ in tqdm(data):
        x, y_true = x[:2], x[2]
        y_pred = model.predict(x)[:, mask_idx]
        y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]]
        y_pred = y_pred.argmax(axis=1)
        y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]])

        right += (y_true == y_pred).sum()
        total += len(y_true)

    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(valid_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights('best_pet_realformer.weights')
        print('acc :{}, best acc:{}'.format(acc, self.best_acc))


if __name__ == '__main__':
    # zero-shot
    zero_acc = evaluate(valid_generator)
    print('zero shot acc: ', zero_acc)
    # few shot
    evaluator = Evaluator()
    train_model.fit_generator(train_generator.forfit(),
                              steps_per_epoch=len(train_generator),
                              epochs=epochs,
                              callbacks=[evaluator])
