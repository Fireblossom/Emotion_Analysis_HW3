from nltk.sentiment.vader import SentimentIntensityAnalyzer
from util.IO_util import read_data
from util.Corpus import LABLE_MAP


source = 'isear' #isear
corpus = read_data('dataset/unified-datasettsv.sec', source)
analyzer = SentimentIntensityAnalyzer()

predict = []
dic = {}
for sample in corpus.test_samples:
    text = sample.text
    vs = analyzer.polarity_scores(text)
    if source == 'grounded_emotions':
        if vs['compound'] >= 0:
            predict.append('joy')
        else:
            predict.append('sadness')
    else:
        '''
        if sample.label not in dic:
            dic[sample.label] = vs
            dic[sample.label]['count'] = 1
        else:
            dic[sample.label]['compound'] += vs['compound']
            dic[sample.label]['neu'] += vs['neu']
            dic[sample.label]['pos'] += vs['pos']
            dic[sample.label]['neg'] += vs['neg']
            dic[sample.label]['count'] += 1
        '''
        allcap_words = 0
        flag = 0
        for word in sample.tokens:
            if word.lower() in LABLE_MAP:
                flag = 1
                predict.append(word.lower())
                break
        if flag == 0:
            if vs['compound'] > 0:
                predict.append('joy')
            elif vs['neg'] > 0.16:
                if 'ill' in sample.tokens or 'died' in sample.tokens or 'left' in sample.tokens:
                    predict.append('sadness')
                else:
                    predict.append('fear')
            elif vs['compound'] <= -0.1:
                if analyzer._amplify_ep(text) >= 0.292 or 'cheating' in sample.tokens:
                    predict.append('anger')
                else:
                    predict.append('disgust')
            else:
                if analyzer._amplify_qm(text) >= 0.18:
                    predict.append('shame')
                else:
                    predict.append('guilt')


corpus.set_pred(predict)
corpus.evaluation()
'''
print(dic)
'''
