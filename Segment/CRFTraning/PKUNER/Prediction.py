from PKUNER.MakeData import LoadData
from PKUNER.MakeMode import sent2features,sent2labels,sent2tokens
import pycrfsuite
import pickle

# def segment(sent,tag):
#     res = ''
#     for i in range(len(sent)):
#         if tag[i] == 'S' or tag[i] == 'E':
#             res+=sent[i]+" "
#         else:
#             res+=sent[i]
#     return res

def prediction(test_file):
    with open(test_file, 'rb') as rp:
        test_set = pickle.load(rp)

    tagger = pycrfsuite.Tagger()
    tagger.open('PKU.crfsuite')

    example_sent = test_set[100]

    # print(example_sent)
    # print(sent2tokens(example_sent))
    # print(' '.join(sent2tokens(example_sent)), end='\n\n')
    # print(sent2features(example_sent))
    # print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    # print("Correct:  ", ' '.join(sent2labels(example_sent)))


def prediction_random(sentence):
    with open(test_file, 'rb') as rp:
        test_set = pickle.load(rp)

    tagger = pycrfsuite.Tagger()
    tagger.open('PKU.crfsuite')

    example = []
    for item in sentence:
        res = [item,'a']
        example.append(res)

    res = tagger.tag(sent2features(example))
    return res
    # print(sent2tokens(example_sent))
    # print(' '.join(sent2tokens(example_sent)), end='\n\n')
    # print(sent2features(example_sent))
    # print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    # print("Correct:  ", ' '.join(sent2labels(example_sent)))

if __name__ == '__main__':
    test_file = 'test.pkl'
    ex = ['（', '苏', '光武', '郭', '全治', '摄影', '报道', '）']
    res = prediction_random(ex)
    # res = prediction_random("不要有列表的嵌套的形式")
    print(res)