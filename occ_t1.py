from nltk.sentiment.vader import SentimentIntensityAnalyzer
from util.IO_util import read_data


corpus = read_data('dataset/unified-datasetjsonl.sec')
analyzer = SentimentIntensityAnalyzer()

scores = []
for sample in corpus.samples:
    text = sample.text
    vs = analyzer.polarity_scores(text)
    scores.append(vs)

##Apply polarity scores to a simple NN maybe
