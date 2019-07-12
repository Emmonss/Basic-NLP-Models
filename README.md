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
and the result of evaluation is following:

| value | precision | recall | f1-score | 
| - | :-: | -: | -: | 
| B | 0.75 | 0.84 | 0.79 |
| M | 0.71 | 0.24 | 0.36 |
| E | 0.78 | 0.88 | 0.80 |
| S | 0.82 | 0.78 | 0.80 |
| Average | 0.78 | 0.78 | 0.78 |


<a name ="crf"></a>
## CRF 条件随机场
1. Usage: need crfsuite
```
  - pip install crfsuite
```
2. Segment and evaluation
  based on file MSRSegment 
  training:
```
  - python TrainingMode.py
```
pay attention to the model_save_path if you want to save model to the path you want.

  predict chinese segment,change the sentence you want in main function
```
  - python Prediction.py
```
  
   Evaluate the model:
```
  -python Evaluation.py
```


| value | precision | recall | f1-score | 
| - | :-: | -: | -: |  
| B | 0.97 | 0.98 | 0.98 |
| M | 0.94 | 0.90 | 0.92 |
| E | 0.97 | 0.98 | 0.97 |
| S | 0.98 | 0.95 | 0.97 |
| Average | 0.97 | 0.97 | 0.97 |


3.  Chinese word Element NER:
  based on file PKUER
  training:
```
  - python MakedataNew.py
  - python TrainingMode.py
```
pay attention to the model_save_path if you want to save model to the path you want.

  predict chinese segment
  I choose one sentence of test data to predict. You can use segment model split sentence first. 
```
  - python Prediction.py
```
  
   Evaluate the model:
```
  - python Evaluation.py
```
the labels is two much so that I do not want to make tabel QAQ.....
  

<a name ="BiLSTMcrf"></a>
## BiLSTM-CRF Segment
Usage: need tensorflow 1.x
change mode in train.py because i did not add args to change mode
you can train model by using Train function
evaluate model by using Test function
predict one sentence by using predict_random function
```
  - python train.py
```
If you think that it is too complicated, you can add args to change it.







