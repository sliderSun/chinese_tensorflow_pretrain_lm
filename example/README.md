# 中文文本分类验证集评测
## 数据说明
|  | 训练集(数量) | 验证集(数量) | 测试集(数量) | 
| :-: | :-: | :-: | :-: | 
| IFLYTEK | 12133 | 2599 | 2600 | 
| TNEWS |  53360 | 10000 | 10000 |   

## 评价指标的说明
- **准确率(ACC)** 
- **F1(宏观|微观)**

## 实验结果: 


### 准确率(ACC):
|  | IFLYTEK | TNEWS | Avg | comment |
| :-: | :-: | :-: | :-: | :-: | 
| baseline (_roberta_) | 60.60 | 57.74 | 59.19 | ** |
| pre_fine (_roberta_) | 61.70 | 58.30 | 60.00 | 预训练+微调 |   
| scl (_roberta_) |  59.94 | 58.08 | 59.01 | Supervised Contrastive Learning |
| grad_penalty (_roberta_) |  62.29 | 58.18 | 60.235 | 梯度惩罚 |  
| adv (_roberta_) | 62.40 | 58.24 | 60.32 | fgm对抗 | 
| attention_adv (_roberta_) | 61.29 | 57.52 | 59.40 | ** | 
| rdrop (_roberta_) | **62.66** | **58.40** | **60.53** | ** |  
| cld (_roberta_) |  ** | ** | ** | 对比学习实验 |  


**注**
- 模型都采用哈工大roberta
 
