from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
import numpy as np

tknzr = TweetTokenizer()


class Corpus:
    def __init__(self):
        self.samples = []
        self.test_samples = []
        self.pred = []
        self.label_set = set()

    def add_sample(self, label, text):
        self.samples.append(Sample(label, text))

    def split_sample(self, ratio=0.1):
        boundary = int(ratio * len(self.samples))
        np.random.shuffle(self.samples)
        self.test_samples = self.samples[:boundary]
        self.samples = self.samples[boundary:]

    def prediction(self, model):
        """
        Call-back style prediction.
        :param model:
        :return:
        """
        for sample in self.samples:
            self.pred.append(model(sample.text))
        self.label_set = self.get_label_set()

    def set_pred(self, pred):
        """
        Make prediction out of object.
        :param pred:
        :return:
        """
        self.pred = pred
        self.label_set = self.get_label_set()

    def get_label_set(self):
        label_set = []
        for sample in self.samples:
            label_set.append(sample.label)
        return set(label_set)

    def evaluation(self):
        if self.pred:
            y_true = []
            for sample in self.samples:
                y_true.append(sample.label)
            t = classification_report(y_true, self.pred, target_names=list(self.label_set))
            print(t)
        else:
            print('Predict firstly.')


LABLE_MAP = {'': 1, 'guilt': 2, 'surprise': 3, 'sadness': 4, 'fear': 5, 'anger': 6, 'noemo': 7, 'trust': 8,
             'love': 9, 'anticipation': 10, 'confusion': 11, 'shame': 12, 'disgust': 13, 'joy': 0}


class Sample:
    def __init__(self, label=None, text=''):
        self.label = label
        self.mapped_label = LABLE_MAP[label]
        self.text = text
        self.tokens = tknzr.tokenize(text)
