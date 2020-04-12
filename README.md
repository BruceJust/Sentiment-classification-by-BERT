# Sentiment-classification-by-BERT
Sentiment classification by BERT on own data



基于自己的数据，使用BERT fine-tuning

该项目使用的外卖的情感分析数据，也在自己的数据上试过，效果不错，供参考。

安装包：

​	1.transformers

​	2.tensorflow

​	3.pytorch（可以不要）

本案例基于huggingface出品的transformers进行BERT fine-tuning。

主要修改glue.py及runtf_glue.py来完成。

详细步骤我在blog写了：

​	

训练结果：

10个epoch

可以看到train_acc一直在增长，val_acc没怎么变，因为已经到88，89了。

BERT模型比较强大，一上来就容易达到最高分了。

![img](https://img-blog.csdnimg.cn/20200412212625319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjE3NTIxNw==,size_16,color_FFFFFF,t_70)