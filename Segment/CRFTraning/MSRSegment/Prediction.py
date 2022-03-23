from MSRSegment.MakeData import LoadData,MakeOneNode
from MSRSegment.MakeMode import sent2features,sent2labels,sent2tokens
import pycrfsuite


def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res

def prediction():
    filename = 'msr_test_gold.utf8'
    test_set = LoadData(filename)

    tagger = pycrfsuite.Tagger()
    tagger.open('msr.crfsuite')

    example_sent = test_set[100]
    print(example_sent)
    print(sent2tokens(example_sent))
    print(' '.join(sent2tokens(example_sent)), end='\n\n')

    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))

def prediction_random(sent):
    example = MakeOneNode(sent)

    tagger = pycrfsuite.Tagger()
    tagger.open('msr.crfsuite')
    res = segment(example,tagger.tag(sent2features(example)))
    return res

if __name__ == '__main__':
    res = prediction_random("不要有列表的嵌套的形式")
    print(res)