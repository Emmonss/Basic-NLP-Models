from MSRSegment.MakeData import LoadData
from MSRSegment.MakeMode import sent2labels,sent2features
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite



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

def Test(filename):

    test_set = LoadData(filename)
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]

    tagger = pycrfsuite.Tagger()
    tagger.open('msr.crfsuite')

    y_pred = [tagger.tag(xseq) for xseq in X_test]

    res = Evaluation(y_test,y_pred)

    print(res)

if __name__ == '__main__':
    filename = '../../datas/msr_data/msr_test_gold.utf8'
    Test(filename = filename)