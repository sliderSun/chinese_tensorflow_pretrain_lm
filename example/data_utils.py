"""
@Time : 2022/9/1 10:00 
@Author : sunshb10145 
@File : data_utils.py
@desc:
"""
import json
import keras
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from bert4keras.snippets import DataGenerator, sequence_padding

from config import maxlen


def load_data(filename, task_name="single", file_type="json", split='\t'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding="utf-8") as f:
        for i, l in tqdm(enumerate(f)):
            if file_type == "json":
                l = json.loads(l)
                if task_name == "iflytek":
                    text, label = l['sentence'], l['label']
                    D.append((text, int(label)))
                elif task_name == "tnews":
                    text, label, label_des = l['sentence'], l['label'], l['label_desc']

                    label = int(label) - 100 if int(label) < 105 else int(label) - 101
                    D.append((text, int(label), label_des))
                elif task_name == "pair":
                    text_a, text_b, label = l['sentence1'], l['sentence1'], l['label']
                    D.append((text_a, text_b, int(label)))
                else:
                    text, label = l['sentence'], l['label']
                    D.append((text, int(label)))
            else:
                if task_name == "iflytek":
                    text, label = l.split(split)
                    D.append((text, int(label)))
                elif task_name == "tnews":
                    text, label, label_des = l.split(split)
                    label = int(label) - 100 if int(label) < 105 else int(label) - 101
                    D.append((text, int(label), label_des))
                elif task_name == "pair":
                    text_a, text_b, label = l.split(split)
                    D.append((text_a, text_b, int(label)))
                else:
                    text, label = l.split(split)
                    D.append((text, int(label)))
    return D


class data_generator(DataGenerator):
    def __init__(self, data, task_name, tokenizer, batch_size=32, buffer_size=None):
        DataGenerator.__init__(self, data, batch_size=batch_size, buffer_size=buffer_size)
        self.task_name = task_name
        self.tokenizer = tokenizer

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if self.task_name == "tnews":
            for is_end, (text, label, label_desc) in self.sample(shuffle):
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)

                    yield [batch_token_ids, batch_segment_ids], batch_labels

                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        elif self.task_name == "pair":
            for is_end, (texta, textb, label) in self.sample(shuffle):
                token_ids, segment_ids = self.tokenizer.encode(texta, textb, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)

                    yield [batch_token_ids, batch_segment_ids], batch_labels

                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for is_end, (text, label) in self.sample(shuffle):
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)

                    yield [batch_token_ids, batch_segment_ids], batch_labels

                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.
        self.best_val_f1_micro = 0.
        self.best_val_f1_macro = 0.

    def evaluate(self, data):
        total, right = 0., 0.
        y_true_list, y_pred_list = list(), list()
        for x_true, y_true in data:
            y_true_list.append(y_true)
            y_pred = self.model.predict(x_true).argmax(axis=1)
            y_pred_list.append(y_pred)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        y_true_np = np.concatenate(y_true_list)
        y_pred_np = np.concatenate(y_pred_list)
        f1_macro = f1_score(y_true_np, y_pred_np, average='macro')
        f1_micro = f1_score(y_true_np, y_pred_np, average='micro')
        return right / total, f1_micro, f1_macro

    def on_epoch_end(self, epoch, logs=None):
        val_acc, f1_micro, f1_macro = self.evaluate(val_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        if f1_micro > self.best_val_f1_micro:
            self.best_val_f1_micro = f1_micro
        if f1_macro > self.best_val_f1_macro:
            self.best_val_f1_micro = f1_macro
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
        print(
            u'f1_micro: %.5f, best_val_f1_micro: %.5f\n' %
            (f1_micro, self.best_val_f1_micro)
        )
        print(
            u'f1_macro: %.5f, best_val_f1_macro: %.5f\n' %
            (f1_macro, self.best_val_f1_macro)
        )
        logs['val_acc'] = val_acc
        logs['f1_micro'] = f1_micro
        logs['f1_macro'] = f1_macro
