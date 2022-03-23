import pickle
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def MakeTag(word):
    if(len(word)==1):
        return 'S'

    else:
        return 'B'+ 'M'*(len(word)-2)+'E'

def Tag2label(tag):
    label = { 'B':0,'M':1,'E':2,'S':3}
    return label[tag]


def Label2Tag(label):
    tag = { 0:'B',1:'M',2:'E',3:'S'}
    return tag[label]

def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res

def LoadMatrix():
    with open('Distribution.pkl', 'rb') as fr:
        ProDic=pickle.load(fr)

    with open('ProDic.pkl', 'rb') as fr:
        Dic=pickle.load(fr)

    return Dic,ProDic


def Evaluation(y_true, y_pred):

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

def makeseq(seq):
    res = ''
    for word in seq.strip().split():
        res+=word
    return res