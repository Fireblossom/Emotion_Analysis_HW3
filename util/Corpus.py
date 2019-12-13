from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report


tknzr = TweetTokenizer()


class Corpus:
    def __init__(self):
        self.samples = []
        self.pred = []

    def add_sample(self, label, text):
        self.samples.append(Sample(label, text))

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


class Sample:
    def __init__(self, label=None, text=''):
        self.label = label
        self.text = text
        self.tokens = tknzr.tokenize(text)
