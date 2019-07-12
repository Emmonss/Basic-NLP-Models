# Chinese Segment and NER Based on Some ML and DL Model
一些中文分词和命名标注的模型与应用

# Tabel of Content

- [HMM]隐马尔科夫模型(#hmm)
- [CRF]条件随机场(#crf)
- [BiLSTM-CRF]双向LSTM-CRF模型(#BiLSTMcrf)

<a name ="hmm"></a>
## HMM 隐马尔科夫分词模型
1. The way to make Dictionary of hmm:<br>
```
  - python CountWords.py
```
and will get two dictionary of hmm: Dic.pkl and ProDic.pkl<br>
2. the way to use hmm to make chinese segment：<br>
```
  - python HMMain.py
```
It is a diplay function, you can change the sentence in main function<br>
3. the way to evaluate the model:<br>
```
  - pyhton Evaluation.py
```
and the result of evaluation is following:<br>
表格
  表头  | 表头
  ------------- | -------------
 单元格内容  | 单元格内容
 单元格内容l  | 单元格内容
