# ASSISTments2009 正式实验协议

冻结日期：2026-11-13。完整命令：

```powershell
.\.venv\Scripts\python.exe -u -m python.run_experiments --config configs/assist2009.yaml --engine cpp --output outputs/assist2009-final
```

若训练进程中断，确认原进程已退出后使用相同命令并追加 `--resume`。程序核对源码摘要、配置、PyTorch 版本和原始数据摘要。已完成的运行复用 `result.json` 和测试预测缓存；未完成训练从最近完整 epoch 的模型、优化器、随机数状态和 DataLoader 状态恢复。

## 数据与预处理

使用 [ASSISTments 官方 Skill-builder 2009–2010 页面](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) 链接的修正版本。原始文件 64,412,812 字节，SHA-256：

```text
162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da
```

发布文件包含非 UTF-8 字节，使用 Latin-1 无丢弃解码。原始 346,860 行中，去除缺失知识点的 63,755 行，再去除交互少于五次的 519 名学生共 1,215 行，最终 281,890 次作答、3,644 名学生。2,021 行耗时被截断至 `[0.5,600]` 秒；修正版在本清洗协议下没有额外重复作答行。

| 集合 | 学生数 | 交互数 | 正例数 | 长于 200 步的学生 |
| --- | ---: | ---: | ---: | ---: |
| Train | 2550 | 202958 | 133085 | 235 |
| Val | 364 | 26425 | 17775 | 26 |
| Test | 730 | 52507 | 34741 | 58 |

学生切分种子为 42。训练词表包含 17,195 道题、149 个概念/复合概念。仅使用训练词表，未见 ID 不进入嵌入词表。OOF 为五折、按学生分配，同一学生所有块保持同折。具体成员与折归属保存在运行目录的 `data_audit.json`，不在论文正文发布逐个学生记录。

该数据集没有可用于本实验的真实作答开始时间，特征 `[9]` 表示同知识点练习步数间隔；`ms_first_response` 用于耗时先验。不能将这项实验的间隔解释为秒级遗忘。

## 固定训练设置

- 种子：42、2025、2026。数据划分与 OOF 划分不随训练种子变化。
- 最大序列长度 200，隐藏维度 64，4 个注意力头，2 层 Transformer，Dropout 0.2。
- Batch size 32；AdamW，学习率 0.001，weight decay 0.0001，梯度裁剪 1.0。
- 最多 30 轮，Val AUC patience=5；仅 Val 参与选模。
- FP32 训练，严格确定性算法；训练 Batch 不按长度重新分组，不采用混合精度；只在无 Dropout 的评估路径裁剪右侧 Padding。
- Windows、Python 3.12、PyTorch 2.14.1+cu130、RTX 3050 Ti Laptop GPU（4GB）。环境版本随 `manifest.json` 保存，另有 `requirements-cuda-lock.txt`。

## 预先确定的实验矩阵

每个种子运行：六模型主实验（StatOnlyLR、DKT、SAKT、CausalSeqOnly、HSF_Concat、HSF_Gate）；HSF_Gate 三组特征消融；Concat/Gate 两个 full_history_carry 变体。共 11×3=33 次独立训练。

主实验均采用 window_aligned。Carry 对比只解释跨块历史摘要作用；表中另列长学生及第 201 步之后的指标，不与主实验同视野增益混为一谈。

每个模型/种子/变体在训练结束并加载最佳验证权重后进行一次 Test 推理，所有分组指标及门控图复用缓存。输出样本数、正例数和 AUC 缺失情况。不会根据 Test 结果改变超参数、替换种子或选择性删除不利模型。

## 结果解释约束

本次只有一个真实数据集。三种子标准差衡量固定划分下的训练波动，不等于跨数据集稳健性、置信区间或统计显著性。真实效果以运行完成后的结果为准，包括无收益或负收益；门控曲线不预设随历史单调下降。

DKT/SAKT 为本项目统一接口的参考实现；StatOnlyLR 带样本内 LayerNorm。HSF 与 SeqOnly 的差异包含显式统计输入及先验校准，不能把二者差值全部归因为门控。Concat 与 Gate 共享统计输入、骨干、校准和参数尺寸，它们之间的受控差值才用于讨论融合方式。
