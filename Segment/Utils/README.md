#分词任务的评估标准<br/>

####1.结果准确率计算<br/>
输出格式如下：<br>
+ y_true<br>
>扬 帆 远 东 做 与 中 国 合 作 的 先 行,B E B E S S B E B E S B E<br>
+ y_pred<br>
>扬 帆 远 东 做 与 中 国 合 作 的 先 行,B E S S S S B E B E S B E<br>

则结果准确率:<br>
<html>
<img src="https://latex.codecogs.com/svg.image?acc&space;=&space;\frac{y^{acc}}{y^{true}}&space;=&space;\frac{11}{13}&space;=&space;84.6%" title="https://latex.codecogs.com/svg.image?acc = \frac{y^{acc}}{y^{true}} = \frac{11}{13} = 84.6%" />
</html>

####2.分词结果的精准率，召回率,F1值<br/>

#### OOV IV 计算<br/>
