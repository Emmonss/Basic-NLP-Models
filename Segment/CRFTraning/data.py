import pandas as pd
from Segment.DataProcess.data import WORD_COL,TAG_COL

def LoadData(filename):
    seq = []
    data = pd.read_csv(filename)
    for index,item in data.iterrows():
        res = []
        words = item[WORD_COL].split()
        tags = item[TAG_COL].split()
        for i in range(len(words)):
            its = []
            its.append(words[i])
            its.append(tags[i])
            res.append(its)
        seq.append(res)
    return seq