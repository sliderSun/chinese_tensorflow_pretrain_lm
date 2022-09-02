"""
@Time : 2022/8/29 22:39 
@Author : sunshb10145 
@File : classification_pretrain_before_finetuning.py 
@desc:
best acc: 0.573 ， 比直接fine-tuning高1.+
环境：tf1.X + tf.keras
"""
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

from tqdm import tqdm
import jieba
import numpy as np

np.random.seed(42)
from bert4keras.models import build_transformer_model, Model
from bert4keras.optimizers import *
from bert4keras.snippets import DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from bert4keras.layers import *
from config import *
from data_utils import load_data, Evaluator

# 加载数据集
train_data = load_data(
    data_dict.get(task_name + "_train"), task_name
)
valid_data = load_data(
    data_dict.get(task_name + "_val"), task_name
)

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

all_data = train_data + valid_data
pretrain_data = [d[0] for d in all_data]

# whole word mask
pretrain_data = [jieba.lcut(d) for d in pretrain_data]


def random_masking(lines):
    """对输入进行随机mask, 支持多行
    """
    if type(lines[0]) != list:
        lines = [lines]

    sources, targets = [tokenizer._token_start_id], [0]
    segments = [0]

    for i, sent in enumerate(lines):
        source, target = [], []
        segment = []
        rands = np.random.random(len(sent))
        for r, word in zip(rands, sent):
            word_token = tokenizer.encode(word)[0][1:-1]

            if r < 0.15 * 0.8:
                source.extend(len(word_token) * [tokenizer._token_mask_id])
                target.extend(word_token)
            elif r < 0.15 * 0.9:
                source.extend(word_token)
                target.extend(word_token)
            elif r < 0.15:
                source.extend([np.random.choice(tokenizer._vocab_size - 5) + 5 for _ in range(len(word_token))])
                target.extend(word_token)
            else:
                source.extend(word_token)
                target.extend([0] * len(word_token))

        # add end token
        source.append(tokenizer._token_end_id)
        target.append(0)

        if i == 0:
            segment = [0] * len(source)
        else:
            segment = [1] * len(source)

        sources.extend(source)
        targets.extend(target)
        segments.extend(segment)

    return sources, targets, segments


class pretrain_data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, = [], [], [], []

        for is_end, item in self.sample(shuffle):
            source_tokens, target_tokens, segment_ids = random_masking(item)

            is_masked = [0 if i == 0 else 1 for i in target_tokens]
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)
            batch_is_masked.append(is_masked)
            #             batch_nsp.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_target_ids = pad_sequences(batch_target_ids)
                batch_is_masked = pad_sequences(batch_is_masked)
                #                 batch_nsp = sequence_padding(batch_nsp)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked], None

                batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked = [], [], [], []


# 补齐最后一个batch
more_ids = batch_size - (len(pretrain_data) % batch_size)
pretrain_data = pretrain_data + pretrain_data[: more_ids]
pretrain_generator = pretrain_data_generator(data=pretrain_data, batch_size=batch_size)


def build_transformer_model_with_mlm():
    """带mlm的bert模型
    """
    bert = build_transformer_model(
        config_path,
        with_mlm='linear',
        #         with_nsp=True,
        model='bert',
        return_keras_model=False,
        #         keep_tokens=keep_tokens
    )
    proba = bert.model.output
    #     print(proba)
    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记

    #     nsp_label = Input(shape=(None, ), dtype='int64', name='nsp')  # nsp

    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def nsp_loss(inputs):
        """计算nsp loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        #         y_pred, _ = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        loss = K.mean(loss)
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    def nsp_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        y_true = K.cast(y_true, K.floatx)
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.mean(acc)
        return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    #     nsp_loss = Lambda(nsp_loss, name='nsp_loss')([nsp_label, proba])
    #     nsp_acc = Lambda(nsp_acc, name='nsp_acc')([nsp_label, proba])

    train_model = Model(
        bert.model.inputs + [token_ids, is_masked], [mlm_loss, mlm_acc]
    )

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
        #         'nsp_loss': lambda y_true, y_pred: y_pred,
        #         'nsp_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


bert, train_model, loss = build_transformer_model_with_mlm()

Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
Opt = extend_with_piecewise_linear_lr(Opt)

opt = Opt(learning_rate=pretrain_lr,
          exclude_from_weight_decay=['Norm', 'bias'],
          grad_accum_steps=2,
          lr_schedule={int(len(pretrain_generator) * pretrain_epochs * 0.1): 1.0,
                       len(pretrain_generator) * pretrain_epochs: 0},
          weight_decay_rate=0.01,
          )

train_model.compile(loss=loss, optimizer=opt)
# 如果传入权重，则加载。注：须在此处加载，才保证不报错。
if checkpoint_path is not None:
    bert.load_weights_from_checkpoint(checkpoint_path)

train_model.summary()

model_saved_path = './bert-wwm-model.ckpt'


class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self):
        self.loss = 1e6

    """自动保存最新模型
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']
            bert.save_weights_as_checkpoint(model_saved_path)

        test_data = all_data[0][0]
        token_ids, segment_ids = tokenizer.encode(test_data)
        token_ids[9] = token_ids[10] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])
        print(tokenizer.decode(probs[0, 9:11].argmax(axis=1)), test_data)


# fine-tune data generator
class data_generator(DataGenerator):
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

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(data=train_data, batch_size=batch_size)
val_generator = data_generator(valid_data, batch_size)

if __name__ == '__main__':
    # pretrain bert use task data
    # 保存模型
    checkpoint = ModelCheckpoint()
    # 记录日志
    csv_logger = keras.callbacks.CSVLogger('training.log')

    train_model.fit(
        pretrain_generator.forfit(),
        steps_per_epoch=len(pretrain_generator),
        epochs=pretrain_epochs,
        callbacks=[checkpoint, csv_logger],
    )

    # build task fine-tune model
    # reload weights without mlm
    # bert_without_mlm = build_transformer_model(checkpoint_path=model_saved_path,
    #                                            config_path=config_path, with_mlm=False)

    idx = 11
    feed_forward_name = 'Transformer-%d-FeedForward' % idx
    bert_without_mlm = bert.layers[feed_forward_name]
    output = Lambda(lambda x: x[:, 0])(bert_without_mlm.output)
    output = Dense(num_classes, activation='softmax')(output)

    model = Model(bert.inputs, output)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(fine_tune_lr),
                  metrics=['sparse_categorical_accuracy'])

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
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=fine_tune_epochs,
        # callbacks=[evaluator],
        callbacks=callbacks
    )
