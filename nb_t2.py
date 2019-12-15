import math

def list_to_dict(lis):
    """
    count the list elements and record them in the dictionary.
    :param lis:
    :return: the dictionary of
    """
    dic = {}
    for elem in lis:
        if elem in dic:
            dic[elem] += 1
        else:
            dic[elem] = 1
    # print(lis)
    # print(dic)
    return dic


def p_tc(dic, term, term_count, b):
    # print((dic[term] + 1) / (term_count + b))
    return (dic[term] + 1) / (term_count + b)


def cmap(pc, p_tc, p_lc):
    """
    Classification rule
    :param pc:
    :param p_tc:
    :param p_lc:
    :return:
    """
    return math.log10(pc) + sum(p_tc) + p_lc


class NB_classifier:
    def __init__(self, features):
        self.model = []
        self.features = features

    def train(self, corpus):
        """
        input a corpus object and train the model.
        :param corpus:
        :return:
        """
        for i in range(self.features):
            positive = []
            positive_count = 0
            negative = []
            negative_count = 0
            for t in range(len(corpus.text)):
                if corpus.gold[t][i] == 1:
                    positive += corpus.text[t].words
                    positive_count += 1
                else:
                    negative += corpus.text[t].words
                    negative_count += 1

            positive_dict = list_to_dict(positive)
            positive_term = sum(positive_dict.values())
            pc_positive = positive_count / positive_count + negative_count
            negative_dict = list_to_dict(negative)
            negative_term = sum(negative_dict.values())
            pc_negative = negative_count / positive_count + negative_count
            # prob of class.

            bins = len(set(list(positive_dict.keys()) + list(negative_dict.keys())))

            p_tc_positive = {}
            for elem in positive_dict:
                p_tc_positive[elem] = p_tc(positive_dict, elem, positive_term, bins)
            p_tc_negative = {}
            for elem in negative_dict:
                p_tc_negative[elem] = p_tc(negative_dict, elem, negative_term, bins)
            # prob of class given term.

            p_lc_positive = {}
            p_lc_negative = {}
            lc_positive = []
            lc_negative = []
            for label in corpus.gold:
                if label[i] == 1:
                    lc_positive.append(tuple(label[:i]))
                else:
                    lc_negative.append(tuple(label[:i]))

            lc_positive_dict = list_to_dict(lc_positive)
            lc_positive_term = sum(lc_positive_dict.values())
            lc_negative_dict = list_to_dict(lc_negative)
            lc_negative_term = sum(lc_negative_dict.values())

            lc_bins = len(set(list(lc_positive_dict.keys()) + list(lc_negative_dict.keys())))

            for elem in lc_positive_dict:
                p_lc_positive[elem] = p_tc(lc_positive_dict, elem, lc_positive_term, lc_bins)
            for elem in lc_negative_dict:
                p_lc_negative[elem] = p_tc(lc_negative_dict, elem, lc_negative_term, lc_bins)
            # prob of class given previous labels sequence.

            dic = {'p_tc_positive': p_tc_positive, 'p_tc_negative': p_tc_negative, 'pc_positive': pc_positive,
                   'pc_negative': pc_negative, 'bins': bins, 'positive_term': positive_term,
                   'negative_term': negative_term, 'p_lc_positive': p_lc_positive, 'p_lc_negative': p_lc_negative,
                   'lc_positive_term': lc_positive_term, 'lc_negative_term': lc_negative_term, 'lc_bins': lc_bins}
            self.model.append(dic)  # Training complete.
            # print(dic)

    def predict(self, corpus):
        """
        input corpus and return the predict labels of given corpus.
        :param corpus:
        :return:
        """
        predict_labels = []
        for index in range(len(corpus.text)):
            word_list = corpus.text[index]
            predict_label = []
            for i in range(self.features):
                p_tc_positive = []
                p_tc_negative = []
                for word in word_list.words:
                    try:
                        p_tc_positive.append(math.log10(self.model[i]['p_tc_positive'][word]))
                    except KeyError:
                        p_tc_positive.append(math.log10(1 / (self.model[i]['positive_term'] + self.model[i]['bins'])))
                    try:
                        p_tc_negative.append(math.log10(self.model[i]['p_tc_negative'][word]))
                    except KeyError:
                        p_tc_negative.append(math.log10(1 / (self.model[i]['negative_term'] + self.model[i]['bins'])))
                if i == 0:
                    p_lc_positive, p_lc_negative = 0, 0
                else:
                    try:
                        p_lc_positive = math.log10(self.model[i]['p_lc_positive'][tuple(predict_label)])
                    except KeyError:
                        p_lc_positive = math.log10(1 / (self.model[i]['lc_positive_term'] + self.model[i]['lc_bins']))
                    try:
                        p_lc_negative = math.log10(self.model[i]['p_lc_negative'][tuple(predict_label)])
                    except KeyError:
                        p_lc_negative = math.log10(1 / (self.model[i]['lc_negative_term'] + self.model[i]['lc_bins']))

                if cmap(self.model[i]['pc_positive'], p_tc_positive, p_lc_positive) > cmap(self.model[i]['pc_negative'], p_tc_negative, p_lc_negative):
                    predict_label.append(1)
                else:
                    predict_label.append(0)
                # print(cmap(self.model[i]['pc_positive'], p_tc_positive, p_lc_positive))
                # print(cmap(self.model[i]['pc_negative'], p_tc_negative, p_lc_negative))
            predict_labels.append(predict_label)

        return predict_labels
