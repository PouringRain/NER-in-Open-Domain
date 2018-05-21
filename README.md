# 腾讯NER项目-实体识别部分

序列化标注工具，实现了Bi-LSTM-CRF模型，并利用pytorch实现了高效的数据加载模块，可以完成:

 - **预处理** 构建词表、label表，从预训练文件构建word embedding，对于不存在于预训练文件中的词，则随机初始化。
 - **训练** 训练模型，保存在开发集上表现最佳的模型。
 - **测试** 加载模型，并对测试文件进行标注输出结果，计算准召率。

## 1. 代码说明

### 1.1 数据格式

数据处理成下列形式，特征与标签之间用空格隔开，标签以BIO的形式进行标注如：B-‘领域标签’，句子之间用一个空行分隔。

      我 O
      们 O
      是 O
      受 O
      到 O
      郑 B-N
      振 I-N
      铎 I-N
      先 O
      生 O
      、 O
      阿 B-N
      英 I-N
      先 O
      生 O
      著 O
      作 O
      的 O
      启 O
      示 O

## 2. 使用说明

### 2.1 训练

语料放在./data目录下，文件名为train.txt。若需要使用预训练的词向量，则添加命令，在对应的文件夹中加入词向量文件，维度为100。

**运行方式:**

    $ python3 train.py --word_dim 100 --word_lstm_dim 256 --pre_emb './embedding/wiki_100.utf8' --dropout 0.2 --use_gpu 0     --gpu_id 0 max_patience 8 --num_epoch 50 --batch_size 16 --use_pretrained_emb 1

**参数说明**

|参数|类型|默认值|备注|
| ------------ | ------------ | ------------ | ------------ |
|word_dim||int|100|词向量的维度|
|use_pretrained_emb|int|1|是否使用预训练的词向量，默认为‘1’，不使用置位‘0’|
|pre_emb|str|'/embedding/wiki_100.utf8'|存放与词典匹配的词向量的文件名|
|word_dim_lstm|int|256|hidden size|
|epochs|int|50|训练的轮数|
|batch_size|int|16|训练时一批数据的样本个数|
|dropout|float|0.2|dropout的概率|
|seed|int|1|随机函数的种子|
|use_gpu|int|0|1表示使用gpu|
|gpu_id|int|0|GPU的卡号，适用于具有多块卡的GPU节点|
|max_patience|int|8|开发集验证模型参数，效果不提升超过8次，训练结束|

运行`python3 train.py -h`可打印出帮助信息。

### 2.2 测试

**运行方式:**

    $ python3 test.py --word_dim 100 --word_lstm_dim 256 --dropout 0.2 --use_gpu 0 --gpu_id 0 --batch_size 128

|参数|类型|默认值|备注|
| ------------ | ------------ | ------------ | ------------ |
|word_dim||int|100|词向量的维度|
|word_dim_lstm|int|256|hidden size|
|batch_size|int|16|训练时一批数据的样本个数|
|eval_batch_size|int|128|测试时一批数据的样本个数|
|dropout|float|0.2|dropout的概率|
|seed|int|1|随机函数的种子|
|use_gpu|int|0|1表示使用gpu|
|gpu_id|int|0|GPU的卡号，适用于具有多块卡的GPU节点|

运行`python3 test.py -h`可打印出帮助信息。

## 3. Requirements
 
 - python==3.6.2
 - numpy==1.13.3
 - torch==0.3.0

## 4. 参考

 - [http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html "http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html")
 - [https://github.com/jiesutd/NCRFpp](https://github.com/jiesutd/NCRFpp "https://github.com/jiesutd/NCRFpp")
 - [http://www.aclweb.org/anthology/N16-1030](http://www.aclweb.org/anthology/N16-1030 "http://www.aclweb.org/anthology/N16-1030")

## 5. TODO
    稍后上传与实体细粒度分类连接代码并更新文档
