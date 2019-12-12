import json
from util.Corpus import Corpus


def read_data(filename):
    corpus = Corpus()
    with open(filename) as file:
        for line in file:
            if line[0] == '{':
                dic = json.loads(line)
                text = dic['text']
                emotions = dic['emotions']
                label = ''
                for emotion in emotions:
                    if emotions[emotion] == 1:
                        label = emotion
                corpus.add_sample(label=label, text=text)
            else:
                lst = line.split('\t')
                text = lst[-2]
                label = lst[-1]
                corpus.add_sample(label=label, text=text)
    return corpus


if __name__ == '__main__':
    read_data('/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Emotion Analysis/Emotion_Analysis_HW3/dataset/unified-datasetjsonl.sec')
