# SQL_dog：从模型训练到产品搭建——迈向Text-to-sql第一步
![SQL_Dog项目介绍](https://raw.githubusercontent.com/qcjySONG/SQL_dog/main/assets/ind.png)
## 介绍

>此为本人的个人项目，完全由本人一个人完成。

此仓库放置了多款模型，这些模型通过在**Qwen3-4B-Instruct-2507**上后训练而来，其主要是服务MIMIC系列数据库的Text-to-Sql任务而生的。其中SQL_Dog_DPO模型为完全由合成数据训练而来的多轮问答模型，企其余模型则为单轮问答而生。具体性能如下：

| 模型名称 (Model) | Baseline (n=1) | ICL  | 自一致性 | **最终执行总分** | **平均消耗 (Tokens/Q)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)** | 41.50 | - | 47.60 (+6.10) | **47.60** | 588.37 |
| **[4B Pro](https://www.modelscope.cn/models/GDUTSONG/4B_Pro)** | 49.40 | - | 60.30 (+10.90) | **60.30** | 359.48 |
| **[4B ProMax](https://www.modelscope.cn/models/GDUTSONG/4B_ProMax)** | 52.30 | - | 63.50 (+11.20) | **63.50** | 365.90 |
| **[30B Coder](https://www.modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct)** | 51.60 | 63.90 (+12.30) | 64.40 (+0.50) | **64.40** | 643.46 |
| **[KIMI K25(Thinking)](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5)** | - | 68.60 (Native) | - | **68.60** | 2,509.44 |
| **[QWEN3.5-27B(Thinking)](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B)** | 52.30 | 66.90 (+14.60) | - | **66.90** | 4,098.57 |
| **[SQL_Dog_DPO](https://www.modelscope.cn/models/GDUTSONG/SQL_Dog_DPO)** | 52.70 | - | 61.00 (+8.30) | **61.00** | 460.91 |

* Note: 4B_Pro和4B_ProMax由[EHRSQL](https://github.com/glee4810/ehrsql-2024)仓库中的训练集经过两轮SFT后训练而来,SQL_Dog_DPO完全由合成数据经过两轮SFT后训练的多轮Text-to-Sql模型。
* 测试集为一个单轮对话数据集，同样来源于[EHRSQL](https://github.com/glee4810/ehrsql-2024)，与之不同的是删除了其中的一些模糊标注。
* 复现：要求完整参考[SQL_Dog数据库](https://github.com/qcjySONG/SQL_dog)

## 表选择器
同样基于Qwen3-4B构建了一个表选择器，其仅需要消耗约10Token（n=1情况下）基于实现对应表级别高召回，具体性能分析如下：

| Method         | Precision (%) | Recall (%) | Strict Recall (%) |
|----------------|---------------|------------|-------------------|
| NaiveTabSelect | 83.33         | 79.92      | 55.53             |
| **[Table Selector](https://www.modelscope.cn/models/GDUTSONG/4B_Table_Selector)** | 80.91         | 98.20      | **95.11**             |

## 训练过程
![幻灯片3](https://raw.githubusercontent.com/qcjySONG/SQL_dog/main/assets/%E5%B9%BB%E7%81%AF%E7%89%873.PNG)

## 合成数据
![幻灯片4](https://raw.githubusercontent.com/qcjySONG/SQL_dog/main/assets/%E5%B9%BB%E7%81%AF%E7%89%874.PNG)

