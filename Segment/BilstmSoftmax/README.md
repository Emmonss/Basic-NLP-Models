# BiLSTM-Softmax for Segment
+ BiLSTM-Softmax 模型的分词运用

## requirement
> pandas<br>
> numpy<br>
> tqdm<br>
> tensorflow=2.4.1<br>

## Input Data Format
+ [how to prepare](./DataProcess/README.md)
> words,tags<br>
扬 帆 远 东 做 与 中 国 合 作 的 先 行,B E B E S S B E B E S B E<br>
希 腊 的 经 济 结 构 较 特 殊 。,B E S B E B E S B E S<br>

## Running Construction
+ ./models
+ ./preds
+ train.py
+ predict.py
+ eval.py

## Data Corpus Evaluation Result
+ 没有训练环境