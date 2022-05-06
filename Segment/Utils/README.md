#分词任务的评估标准<br/>

####1.结果准确率计算<br/>
输出格式如下：<br>
+ y_true<br>
>扬 帆 远 东 做 与 中 国 合 作 的 先 行,B E B E S S B E B E S B E<br>
+ y_pred<br>
>扬 帆 远 东 做 与 中 国 合 作 的 先 行,B E S S S S B E B E S B E<br>

则结果准确率:<br>
$acc = (y_true&y_pred)/y_pred = 11/13 = 84.6%$

####2.分词结果的精准率，召回率,F1值<br/>

#### OOV IV 计算<br/>
