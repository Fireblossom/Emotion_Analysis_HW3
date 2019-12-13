import json
from util.Corpus import Corpus


def read_data(filename, source='tec'):
    corpus = Corpus()
    with open(filename) as file:
        for line in file:
            if line[0] == '{':
                dic = json.loads(line)
                if dic['labeled'] == 'single':
                    text = dic['text']
                    emotions = dic['emotions']
                    label = ''
                    for emotion in emotions:
                        if emotions[emotion] == 1:
                            label = emotion
                    if label == '':
                        print(text)
                    corpus.add_sample(label=label, text=text)
            else:
                lst = line.split('\t')
                if lst[1] == source:
                    text = lst[-2]
                    label = lst[-1][:-1]
                    corpus.add_sample(label=label, text=text)
    corpus.split_sample()
    return corpus


if __name__ == '__main__':
    corpus = read_data('/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Emotion Analysis/Emotion_Analysis_HW3/dataset/unified-datasetjsonl.sec',
                       'isear')# grounded_emotions
    print(corpus.get_label_set())
