# 《融合历史统计特征的知识追踪方法研究》完整工程设计书与 GPT 开发手册（严谨定稿版）

> **定稿说明（已全面吸收方法论评审意见）**：  
> 本设计书将论文核心科学问题聚焦于：**“给定学生即将作答的题目，零泄露多尺度显式历史统计特征能否提升答题正确率预测？这种增益在学生作答历史较短或题目低频稀疏时是否更显著？在严格同输入、同视野、同偏置条件下，自适应门控融合相比直接拼接是否有额外收益？”**  
> 本版已彻底修补四处关键方法论细节：**① 训练集学生级 5 折折外（Out-of-Fold）先验拟合防止自泄露；② 主实验统一块内历史视野防止跨块记忆混杂归因；③ 严格区分间隔步数与真实时间戳、区分短历史学生与低频题目；④ 采用“纯统计逻辑回归 → 同骨干纯序列 → 同骨干拼接 → 同骨干门控”四阶受控对照 + DKT/SAKT 经典参照**，确保实验结论无懈可击、完全经得起答辩追问。

---

## 第 0 章 5 分钟全局人话导读（给你自己看的通俗说明书）

### 0.1 整篇论文在回答哪三个最核心的问题？
1. **统计特征本身有多大威力、又能帮序列神经网络补多少短板？**  
   以前做知识追踪（预测学生下一题能不能做对），大家习惯直接上复杂的神经网络（如 `DKT`、`SAKT`）。但新学生刚做前几道题（**短历史冷启动**）或遇到题库里的偏门题（**低频稀疏题**）时，神经网络学不到稳定的 ID 向量。我们提取三层 10 维直观的“历史统计特征”（题目难度、学生总正确率、近期 5 题正确率、连对连错、间隔步数/时间），先测光靠逻辑回归（`Stat-Only LR`）能考多少分，再看加进序列神经网络（`Seq + Stat`）能涨多少分！
2. **门控融合（Gate）真的比直接拼接（Concat）更好吗？**  
   为了不“作弊偏心”，我们让“直接拼接”和“自适应门控”用**完全一模一样的特征、一模一样的历史长度、一模一样的题目难度偏置**公平PK。如果门控赢了，说明动态调节权重确实更优；哪怕两者在整体指标上接近，门控在冷启动前几题的自动倾斜规律本身也是非常诚实、有说服力的可解释性分析！
3. **C++17 引擎在工程上解决了什么真问题？**  
   为了防止训练集自己偷看自己的答案，训练集必须做 **5 折交叉折外统计**，还要对成千上万名学生的长序列一步步递推滑动窗口与哈希状态，纯 Python 循环很慢。我们用纯标准库 C++17 手写**泛型定长环形队列 `RingBuffer<T>`** 与**多线程池 `ThreadPool`**，编成动态库给 Python 内存直连调用，实实在在把特征构建速度提升数十倍！

### 0.2 刚才评审指出的 4 个“防翻车细节”用大白话怎么理解？
- **细节 1（折外先验防自泄露）**：假设某道偏门题在训练集只被张三做过 1 次且做对了。如果用整个训练集算这道题的正确率再喂给张三训练，模型就会发现“只要这道题先验偏高，答案就是对”——这叫**训练集目标自泄露**！解决办法超简单：把训练集学生分成 5 份，给第 1 份学生算题目难度时只用第 2～5 份学生的数据（叫**折外统计 Out-of-Fold**）；而验证集和测试集学生则正常用全部训练集算好的难度。
- **细节 2（统一历史视野防胜之不武）**：如果一个学生做了 350 题，神经网络最多看 200 题，切到第 201～350 题时神经网络从头开始看，统计特征却带着前 200 题的家底，那赢了也不公平。所以我们**主实验统一在相同的 200 题视野内公平对比**；另外单独加一个小开关测“跨块携带统计摘要”能帮长序列补多少分，两边都讲得明明白白！
- **细节 3（诚实命名时间特征）**：`ASSISTments2009` 数据集只有做题先后序号，没有真实的几点几分时间戳，算出来的是**“同知识点间隔了多少道题（练习步数间隔）”**，不能瞎叫“隔了多少秒”；只有换到带真实时间戳的 `ASSISTments2017` 时才叫**“真实时间间隔”**。
- **细节 4（严格锁死测试集）**：必须按学生切分 `Train (70%) / Val (10%) / Test (20%)`，训练时每轮只看 `Val AUC` 挑最好的模型存档，最后**只跑一次 `Test` 集出论文表格**，绝不边看测试集边调参！

---

## 第一章 项目完整目录结构与全文件清单（共 20 个文件）

目录扁平清晰，无多余嵌套，AI 必须严格按此路径与文件名输出：

```text
Historical-Statistics-Knowledge-Tracing/
├── CMakeLists.txt                      # [1] C++17 跨平台构建配置 (生成 hsf_stat 动态库 + C++单测压测程序)
├── requirements.txt                    # [2] Python 依赖包版本清单
├── README.md                           # [3] 编译指南、数据准备说明与严谨复现手册
├── configs/
│   └── default.yaml                    # [4] 数据切分、OOF折数、模型超参与实验控制统一配置
├── cpp/                                # ========= Modern C++17 高性能并发特征引擎 (8 个文件) =========
│   ├── include/hsf/
│   │   ├── types.hpp                   # [5] Interaction / StatConfig 结构体、常量与错误码枚举
│   │   ├── ring_buffer.hpp             # [6] 泛型定长环形缓冲区模板 RingBuffer<T> (O(1) 滑动窗口均值)
│   │   ├── thread_pool.hpp             # [7] 标准库 C++17 线程池 ThreadPool (RAII 多核并行分片调度)
│   │   ├── stat_engine.hpp             # [8] StudentState 状态机、HistoricalStatEngine 与 OOF 批量提取接口
│   │   └── c_api.h                     # [9] 供 Python ctypes 零拷贝调用的标准 C 导出接口头文件
│   ├── src/
│   │   ├── stat_engine.cpp             # [10] 三层 10 维特征提取、OOF 折外先验拟合与多线程实现
│   │   └── c_api.cpp                   # [11] extern "C" 动态库实现 (内置 try-catch 异常安全防火墙)
│   └── apps/
│       └── bench_and_test.cpp          # [12] C++ 原生单测 (步内防泄露 + OOF 防自泄露) 与多核吞吐压测工具
├── python/                             # ========= PyTorch 受控对照实验与可视化系统 (7 个文件) =========
│   ├── config.py                       # [13] @dataclass(frozen=True) 强类型配置类与 YAML 校验加载器
│   ├── cpp_bridge.py                   # [14] ctypes 动态库封装器 + 公式逐位一致的纯 Python 对照提取器
│   ├── data_pipeline.py                # [15] 真实数据集解析、仿真数据、学生级切分、OOF 特征构建与 Dataset
│   ├── models.py                       # [16] 六大对照模型: StatOnlyLR, DKT, SAKT, SeqOnly, HSF_Concat, HSF_Gate
│   ├── trainer.py                      # [17] 严格 Train/Val 早停训练器 + 短历史步数分桶 & 低频题分桶评估器
│   ├── run_experiments.py              # [18] 一键运行主实验、三类特征消融、视野对比与 C++ 加速评测并导表
│   └── plot_figures.py                 # [19] 自动绘制 5 张 300-DPI 论文高清插图 (.png + .pdf)
└── tests/
    └── test_rigor.py                   # [20] pytest 严谨性验收测试 (OOF 零自泄露、步内零泄露、C++/Py 对齐、同视野对齐)
```

---

## 第二章 代码规范硬性契约（Clean Code + Effective C++ + Google Style）

### 2.1 C++17 规范（Google C++ Style + *Effective C++* + *Clean Code*）
1. **命名规则**：统一置于 `namespace hsf`；类型与函数使用 `PascalCase`；局部变量与形参使用 `snake_case`；类私有成员变量必须以单下划线结尾（如 `window_buf_`）；编译期常量以 `k` 开头接 `PascalCase`（如 `constexpr std::size_t kNumStatFeatures = 10;`），严禁用 `#define` 定义常量。
2. **RAII 与拷贝控制（*Effective C++ Item 13, Rule of Five*）**：严禁裸 `new`/`delete`；`ThreadPool` 等管理线程/锁资源的类必须显式声明拷贝构造与拷贝赋值为 `= delete`，析构函数自动安全唤醒并 `join` 全部线程。
3. **`const`、`explicit` 与零警告纪律**：单参数构造函数一律加 `explicit`；只读成员函数加 `const`，不抛异常的计算函数加 `noexcept`；成员初始化顺序与声明顺序严格一致；在 Windows MinGW (`g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic`) 与 Linux GCC 下**零警告编译通过**。
4. **C ABI 边界异常安全**：`c_api.cpp` 所有 `extern "C"` 函数内部必须用 `try-catch` 拦截全部异常并返回负值错误码，严禁异常穿透动态库边界。

### 2.2 Python 规范（Google Python Style + *Clean Code*）
1. **静态类型与冻结配置**：首行统一 `from __future__ import annotations`，全部函数具备完整 Type Hints；使用 `@dataclass(frozen=True)` 管理配置，禁止传递无结构字典。
2. **短函数与反偷懒红线**：遵循单一职责原则，单函数不超过 45 行；**严禁出现任何 `TODO`、`FIXME`、`pass`、`...` 或省略代码**。
3. **张量维度行尾强制注释**：`models.py` 中每一次矩阵/张量运算必须在行尾注释精确形状（如 `# [B, L, D]`）。

---

## 第三章 严谨数据协议与三层 10 维统计特征规格（核心定稿章）

### 3.1 数据集元数据标记与清洗切分协议（`python/data_pipeline.py`）
每个数据集加载后封装为 `DatasetBundle`，显式携带两个诚实布尔标记：`has_real_timestamp: bool` 与 `has_real_resp_time: bool`：
1. **`ASSISTments2009`**（`skill_builder_data.csv`）：
   - `has_real_timestamp = False`, `has_real_resp_time = True`（若有 `ms_first_response` 列则转秒并截断至 `[0.5, 600.0]`，若当前版本 CSV 无该列则自动置 `has_real_resp_time = False`）。
   - 过滤 `skill_id` 为空的记录；按 `(user_id, order_id)` 稳定排序；`timestamp_sec` 字段直接填入单调递增步序号 `0.0, 1.0, 2.0, ...`（**注意：此时特征引擎按 `has_real_timestamp = False` 分支计算“同知识点间隔步数”，绝不乘 60 伪造秒数！**）。
2. **`ASSISTments2017`**（`anonymized_full_release_competition_dataset.csv`）或 **`EdNet-KT1`**：
   - `has_real_timestamp = True`, `has_real_resp_time = True`，读取真实秒级 `startTime` 与 `timeTaken`（截断至 `[0.5, 600.0]`）。
3. **内置仿真数据集 `synthetic`**：
   - `has_real_timestamp = True`, `has_real_resp_time = True`，按 IRT 题目难度 + 学生概念熟练度增长 + 时间遗忘衰减生成 `1200` 名学生数据，专供无外部 CSV 时快速跑通全流程闭环测试与单测。
4. **ID 重映射与严格按学生三划分**：
   - 过滤总交互步数 `< 5` 的学生；`0` 严格预留给 Padding，有效题目重映射为 `1 .. num_items`，有效知识点重映射为 `1 .. num_concepts`（仅根据 `Train` 集建立词表，验证/测试集若遇未见新题或新概念，在统计引擎中按未见 ID 平滑回退至全局均值 $\bar{r}$，在 Embedding 层映射至专用的 `<UNK>` 索引 `num_items + 1`！这一点彻底杜绝测试集词表越界或泄露！）。
   - 按 `user_id` 固定随机种子切分为 **Train (70%) / Val (10%) / Test (20%)**，三集合学生交集严格为空。

### 3.2 消除自泄露的折外先验拟合协议（Out-of-Fold Prior Fitting）
1. **训练集内部 5 折折外统计（OOF）**：
   - 将 `Train` 集学生均匀随机分为 $K_{\text{oof}} = 5$ 折（`Fold 0 .. 4`）。
   - 对属于 `Fold m` 的训练学生：仅用其余 4 折（`Train \ Fold m`）的全部交互调用 `engine.FitOnTrain(...)` 统计全局均值 $\bar{r}^{(-m)}, \bar{\tau}^{(-m)}$ 及各题目/知识点的正确率与平均耗时，再提取 `Fold m` 学生的 10 维特征！
   - 这样即便某道低频题在 `Fold m` 中只出现 1 次，该样本自身的答案 $r_t$ 也**绝对不会进入它自己的 `item_prior_acc`**！
2. **验证集与测试集统计**：
   - 使用 **100% 完整 `Train` 集**调用一次 `full_engine.FitOnTrain(all_train_seqs)`，再用 `full_engine` 提取 `Val` 集和 `Test` 集的特征，同时保存每道题在完整训练集中的出现频次 `train_item_freq[item_id]`（供测试集低频题分组评估使用）。

### 3.3 统一历史视野分块协议（消除跨块记忆归因混杂）
设置单块最大序列长度 `max_seq_len = 200`，提供两种显式受控模式（通过配置 `chunk_protocol` 切换）：
1. **`"window_aligned"`（主实验默认公平模式）**：
   - 先将超长学生序列按 `max_seq_len = 200` 切分为独立窗口块，然后在**每个块内部从局部第 0 步开始调用统计引擎递推提取个人历史统计特征**，并标记块内局部步号与原序列真实步号。
   - **科学意义**：在此模式下，同一分块内纯序列模型（`SeqOnly` / `SAKT` / `DKT`）、拼接模型（`HSF_Concat`）和门控模型（`HSF_Gate`）看到的个人历史交互集合 **100% 完全相同**！测出的任何性能提升均纯粹来自显式统计特征表示，绝无跨块历史记忆干扰！
2. **`"full_history_carry"`（单独附加实验模式）**：
   - 先在学生未切分的完整长序列（如 350 步）上一次性提取个人统计特征，再切分为 200 步的块。专门用于在附加小节验证：“当 Transformer 上下文窗口受限时，轻量级 `O(1)` 历史统计状态能否低成本补充跨块长程记忆”。

### 3.4 三层 10 维历史统计特征精确数学定义（`kNumStatFeatures = 10`）
默认平滑超参数：$\alpha_p = 10.0$，$\alpha_u = 2.0$，滑动窗口大小 $K = 5$。  
对于序列内第 $t$ 步（$t = 0, 1, \dots, L-1$），待作答交互为 $(q_t, c_t, \text{ts}_t)$，**在读取真实标签 $r_t$ 与耗时 $\tau_t$ 之前**计算：

| 组别 | 维度 | 特征名称 | 精确公式与数据集自适应行为 | $t=0$ 或未见 ID 边界值 |
| :--- | :---: | :--- | :--- | :--- |
| **A. 题目与概念先验层**<br>（OOF / Train 拟合） | `[0]` | `item_prior_acc` | $\frac{C_{q_t} + \alpha_p \bar{r}}{N_{q_t} + \alpha_p}$ | 未见题目 $N_{q_t}=0$ 时精确等于 $\bar{r}$ |
| | `[1]` | `concept_prior_acc` | $\frac{C_{c_t} + \alpha_p \bar{r}}{N_{c_t} + \alpha_p}$ | 未见概念 $N_{c_t}=0$ 时精确等于 $\bar{r}$ |
| | `[2]` | `item_mean_rt_norm` | 若 `has_real_resp_time` 为真：$\frac{\ln(1 + \hat{\tau}_{q_t})}{6.0}$（其中 $\hat{\tau}_q = \frac{S_q + \alpha_p \bar{\tau}}{N_q + \alpha_p}$）；否则恒为 `0.0f` | 无耗时数据集自动置 `0.0f`，不引入假常数噪声 |
| **B. 学生长期累计层**<br>（`0..t-1` 因果累加） | `[3]` | `user_global_acc` | $\frac{\text{u\_cor}_{<t} + \alpha_u \bar{r}}{t + \alpha_u}$ | $t=0$ 时精确等于 $\bar{r}$ |
| | `[4]` | `user_total_cnt_norm`| $\frac{\ln(1.0 + t)}{6.0}$ | $t=0$ 时精确为 `0.0f` |
| | `[5]` | `user_concept_acc` | $\frac{\text{uc\_cor}_{c_t, <t} + \alpha_u \cdot \text{feat}[1]}{\text{uc\_cnt}_{c_t, <t} + \alpha_u}$ | 首次遇到概念 $c_t$ 时等于概念先验 `feat[1]` |
| | `[6]` | `user_concept_cnt_norm`| $\frac{\ln(1.0 + \text{uc\_cnt}_{c_t, <t})}{5.0}$ | 首次遇到概念 $c_t$ 时精确为 `0.0f` |
| **C. 近期状态与间隔层**<br>（窗口 & 间隔） | `[7]` | `window_acc_k` | `RingBuffer` 内最近 $\min(t, K)$ 步正确率均值 | $t=0$ 窗口为空时等于 $\bar{r}$ |
| | `[8]` | `signed_streak_tanh` | $\tanh(\text{streak}_{<t} / 5.0)$（连对正累加，连错负累加） | $t=0$ 时为 `0.0f` |
| | `[9]` | `concept_interval_norm`| 若此前练过概念 $c_t$：<br>• `has_real_timestamp=True`（真实秒差）：$\frac{\ln(1 + \text{clamp}(\Delta \text{ts}, 0, 2592000))}{15.0}$<br>• `has_real_timestamp=False`（步数间隔）：$\frac{\ln(1 + \min(t - \text{last\_step}_{c_t}, 500))}{6.5}$ | 首次遇到概念 $c_t$ 时精确为 `0.0f` |

计算完第 $t$ 步的 10 维向量后，再用真实答案 $r_t$ 更新学生累计计数、概念计数、`RingBuffer`、连击数与该概念最后出现位置/时间戳。

---

## 第四章 C++17 高性能并发特征引擎详细设计

1. **`cpp/include/hsf/types.hpp`**：定义 `kNumStatFeatures = 10`、`StatusCode`、`Interaction` 以及包含 `bool has_real_timestamp = true; bool has_real_resp_time = true;` 的 `StatConfig`。
2. **`cpp/include/hsf/ring_buffer.hpp`**：泛型 `RingBuffer<T>`，`O(1)` 时间维护定长窗口滚动和与均值，无运行期堆分配。
3. **`cpp/include/hsf/thread_pool.hpp`**：标准 C++17 `ThreadPool`，RAII 自动回收线程，支持指定任意线程数并发执行分片任务。
4. **`cpp/include/hsf/stat_engine.hpp` & `src/stat_engine.cpp`**：
   - 提供标准 `FitFlat(...)` 与多线程 `TransformFlat(..., size_t num_threads, float* out_features)`；
   - **原生内置高速 OOF 训练集提取函数 `FitAndTransformOOF(...)`**：接收训练集全部序列与每条序列所属的 `fold_ids[num_seqs]`（`0 .. num_folds-1`），在 C++ 内部高效完成“按折汇总全局计数 → 减去当前折计数即得其余 $K-1$ 折精确先验（`O(1)` 作差法，无需重复遍历 $K$ 遍数据！）→ 多线程并行提取全部训练样本折外特征”！
   > **工程亮点提示**：用**“总计数减去本折计数”的 `O(1)` 差分法**实现 K 折折外先验，不仅数学上与重跑 $K$ 次完全等价，而且时间复杂度从 $O(K \cdot N)$ 降到了 $O(N)$！答辩时讲出这个算法优化细节，评委绝对赞不绝口！
5. **`cpp/include/hsf/c_api.h` & `src/c_api.cpp`**：导出 `hsf_create_engine`, `hsf_destroy_engine`, `hsf_fit_flat`, `hsf_transform_flat`, `hsf_fit_transform_oof_flat` 五个异常安全的 C 函数供 Python `ctypes` 直接内存调用。

---

## 第五章 六大严谨受控对照模型架构设计（`python/models.py`）

为彻底回答“到底是哪一部分带来了提升”，设计 **4 个严格控制变量的阶梯模型 + 2 个经典文献参照模型**。所有序列模型在 `forward` 入口统一执行历史交互右移 1 位（`prev_q, prev_c, prev_r`），物理杜绝读取当前步标签：

1. **`StatOnlyLR`（纯统计特征逻辑回归基线，对标 Gervet et al., EDM 2020）**：
   - **完全不使用任何 ID Embedding 或序列神经网络**！
   - 仅将当前步 10 维统计特征 $\mathbf{s}_t$ 加上题目与概念先验对数赔率（$\text{logit}(s_{t,0})$ 与 $\text{logit}(s_{t,1})$）共 12 维数值，经 `LayerNorm(12) -> Linear(12, 1)` 直接输出预测 `logits: [B, L]`。用于直接衡量显式历史统计特征自身的基线预测力！
2. **`DKT`（经典循环神经网络参照，Piech et al., NeurIPS 2015）**：
   - 历史交互嵌入输入单向 `nn.GRU(D, D)`，拼接当前目标题目与概念嵌入经线性层预测 `logits`。
3. **`SAKT`（经典自注意力网络参照，Pandey & Karypis, EDM 2019）**：
   - 以目标题目/概念嵌入为 Query、右移历史交互嵌入为 Key/Value，经单向因果掩码多头注意力与残差 FFN 输出 `logits`。
4. **`CausalSeqOnly`（同骨干纯序列对照模型）**：
   - 使用与本文主模型**完全相同的 2 层因果掩码 TransformerEncoder 骨干**提取序列隐状态 $\mathbf{h}_t: [B, L, D]$，直接与目标题表征 $\mathbf{q}^{\text{tar}}_t = \text{item\_emb}(q_t) + \text{concept\_emb}(c_t)$ 拼接送入双层 `pred_head` 输出 `logits`（不使用任何统计特征，不加先验偏置）。
5. **`HSF_Concat`（同骨干、同特征、同偏置直接拼接对照模型）**：
   - 在 `CausalSeqOnly` 基础上，用 `ContinuousStatEncoder`（`LayerNorm(10) -> Linear(10, D) -> GELU -> Dropout -> Linear(D, D) -> LayerNorm(D)`）将 10 维统计特征编码为 $\mathbf{e}_t: [B, L, D]$；
   - **直接拼接融合**：$\mathbf{z}_t = \text{Linear}_{2D \to D}([\mathbf{h}_t \parallel \mathbf{e}_t])$；
   - 与目标题表征 $\mathbf{q}^{\text{tar}}_t$ 拼入 `pred_head` 得 `raw_logits`，并**加上与门控版完全一致的可学习题目先验对数赔率校准项** $\lambda \cdot \ln\frac{\text{clamp}(s_{t,0}, 10^{-3}, 0.999)}{1 - \text{clamp}(s_{t,0}, 10^{-3}, 0.999)}$！
6. **`HSF_Gate`（本文完整自适应门控融合模型）**：
   - 与 `HSF_Concat` 共享**100% 相同的骨干、相同的统计编码器、相同的历史视野与相同的先验对数赔率校准项**，唯一区别是将融合层替换为自适应门控：
     $$\mathbf{g}_t = \sigma\left(\mathbf{W}_g [\mathbf{h}_t \parallel \mathbf{e}_t] + \mathbf{b}_g\right), \quad \mathbf{z}_t = \mathbf{g}_t \odot \mathbf{h}_t + (1 - \mathbf{g}_t) \odot \mathbf{e}_t$$
   - 同时返回门控张量 $\mathbf{g}_t: [B, L, D]$ 供早期冷启动权重演化分析，并支持特征组消融掩码（`ablate_group = "none" | "item_prior" | "user_cum" | "recent_state"`，分别将对应组维度 `[0:3]`、`[3:7]`、`[7:10]` 置零；注意当消融 `"item_prior"` 时同步关闭先验对数赔率校准项！）。

---

## 第六章 训练评估规范、五大实验矩阵与 5 张论文插图规格

### 6.1 严格无泄露训练与多维分组评估（`python/trainer.py`）
- **选模准则**：每个 Epoch 结束仅在 **Validation 集**上计算 `Val AUC`；使用 `EarlyStopping(patience=5)` 保存 `Val AUC` 最高的模型检查点；训练彻底终止后，加载最优验证集检查点在 **Test 集上仅执行一次最终评测**！支持多随机种子（如 `seeds = [42, 2025, 2026]`）自动计算 `Mean ± Std`。
- **Test 集输出指标全集**：
  1. **全局主指标**：`overall_auc`, `overall_acc`, `overall_rmse`；
  2. **按学生历史步数分组 AUC（验证短历史冷启动增益）**：`step_1_5_auc`, `step_6_10_auc`, `step_11_30_auc`, `step_gt_30_auc`；
  3. **按题目训练集频次分组 AUC（验证低频稀疏题增益）**：根据 `train_item_freq[q]` 统计 `freq_0_unseen_auc`（若有）、`freq_1_5_rare_auc`、`freq_6_20_med_auc`、`freq_gt_20_common_auc`。

### 6.2 自动运行的五大核心实验（`python/run_experiments.py`）
1. **实验一（受控主实验对比）**：在统一历史视野（`window_aligned`）下对比 `StatOnlyLR`, `DKT`, `SAKT`, `CausalSeqOnly`, `HSF_Concat`, `HSF_Gate` 的全局指标（`Mean ± Std`），清晰量化“纯统计有多强”“统计给序列补了多少”“门控比拼接多了多少”。
2. **实验二（学生短历史冷启动 & 低频题双分组分析）**：对比 `CausalSeqOnly`, `HSF_Concat`, `HSF_Gate`, `SAKT` 在 4 个学生步数区间和题目频次区间上的 AUC 增益。
3. **实验三（三层统计特征消融实验）**：基于 `HSF_Gate` 分别移除 A 组（题目与概念先验）、B 组（学生长期累计）、C 组（近期窗口与间隔），检验每一层特征的独立贡献。
4. **实验四（分块视野归因对比实验）**：针对长度 `> 200` 的长序列学生，对比 `window_aligned` 与 `full_history_carry` 两种模式，诚实分离“同视野特征增益”与“跨块长程统计记忆增益”。
5. **实验五（C++17 引擎 OOF + 多线程加速评测）**：在同一真实/仿真数据集规模（及扩展规模 `50K / 200K / 1M` 交互）上，实测纯 Python vs C++17（1 / 2 / 4 / 8 线程）执行 5 折 OOF 拟合与特征提取的真实耗时（ms）与加速比，并验证数值误差 `< 1e-6`。

### 6.3 自动生成的 5 张 300-DPI 论文插图（`python/plot_figures.py`）
1. `fig1_controlled_ladder.{png,pdf}`：六大对照模型 Overall AUC 与前 5 题 / 前 10 题 AUC 阶梯对比柱状图（含误差棒）；
2. `fig2_short_history_and_rarity.{png,pdf}`：左子图为学生作答步数分桶 AUC 折线图，右子图为题目训练频次分桶 AUC 柱状图；
3. `fig3_feature_ablation.{png,pdf}`：三组统计特征独立消融对整体 AUC 与早期 AUC 的影响对比图；
4. `fig4_gate_dynamics.{png,pdf}`：测试集学生随作答步数从第 1 题推进到第 50 题时，统计分支平均门控权重 $(1 - \bar{g}_t)$ 的群体均值带（Mean ± SD 阴影曲线）与典型个案轨迹（直观展示“前几题更依赖统计先验，随历史增长逐步转向序列与统计协同”）；
5. `fig5_cpp_oof_speedup.{png,pdf}`：Python vs C++17（1/2/4/8 线程）执行 OOF 特征提取的耗时与加速比对比图。

---

## 第七章 四项严谨性自动化测试（`tests/test_rigor.py` & `bench_and_test.cpp`）

1. **`test_oof_no_self_leakage`（折外先验防自泄露测试）**：构造含大量只出现 1 次孤立题目的训练集，任意挑选一名训练学生将其所有 `correct` 从 `0` 改为 `1`（或取反），重新运行 OOF 特征提取，断言**该学生自身序列上所有步的 `item_prior_acc` 与 `concept_prior_acc` 差值严格等于 `0.0`**！
2. **`test_causal_step_invariance`（步内因果防未来泄露测试）**：翻转任意测试序列第 $t$ 步及之后的全部答案与耗时，断言特征提取器与全部 6 个模型在第 $0 \dots t$ 步（含第 $t$ 步）的特征和输出概率差值 `< 1e-6`！
3. **`test_cpp_python_oof_and_transform_parity`（跨语言逐位对齐测试）**：验证 C++ 差分法 OOF 提取及普通批量提取（1 线程与 4 线程）与纯 Python 实现输出矩阵的最大绝对误差 `< 1e-6`！
4. **`test_window_aligned_fair_horizon`（同视野公平性测试）**：验证在 `window_aligned` 模式下，学生第 2 个切分块的第 0 步个人统计特征已按块内局部历史重置（`user_total_cnt_norm == 0.0`），绝不偷偷携带上一块的个人累计计数。

---

## 第八章 交给 GPT-Plus 的 4 步分批开发指令集（直接复制使用）

> **投喂说明**：在 GPT-Plus 对话中，先把上面 **第 0 章～第 7 章全文 + 下方【Step 0】** 发给它完成定稿确认；随后依次发送 **【Step 1】→【Step 2】→【Step 3】→【Step 4】**，即可完整拿回全部 20 个零省略、方法论严谨的源码文件！

### 【Step 0：发送修订定稿版设计书 + 锁定契约】
将本文档第 0～7 章全文粘贴给 GPT，并在末尾附上：
```text
我已经采纳了之前的全部评审建议，将第 3 章的数据防泄露协议（训练集学生级 5 折 OOF 先验拟合、主实验 window_aligned 统一块内历史视野、真实时间戳与练习步数间隔诚实区分、<UNK> 未见词表安全映射）以及第 5~6 章的六模型受控阶梯对照方案全部修订定稿如上。
请你仔细阅读并牢记本定稿规格书的全部细节与编码规范（Google Style + Clean Code + Effective C++）。本轮请不要输出代码，仅简要确认核心契约已锁定并回复 "RIGOR_SPEC_READY"，随后我分 4 批让你输出全部 20 个文件的 100% 完整源码（严禁任何 TODO 或省略）。
```

### 【Step 1：交付全部 C++17 引擎、O(1) 差分 OOF 算子与单测压测源码（第 1 批：9 个文件）】
```text
现在执行【第 1 批交付】：请严格按照定稿设计书第 1~4 章规格及 Google C++ / Effective C++ 规范（RAII、Rule of Five、const/explicit/noexcept、C-ABI try-catch 防火墙、Windows MinGW 与 Linux GCC 双平台 `-O3 -Wall -Wextra -Wpedantic` 零警告），输出以下 9 个文件的 100% 完整代码（严禁任何省略）：
1. CMakeLists.txt
2. cpp/include/hsf/types.hpp (含 has_real_timestamp 与 has_real_resp_time 配置位)
3. cpp/include/hsf/ring_buffer.hpp
4. cpp/include/hsf/thread_pool.hpp
5. cpp/include/hsf/stat_engine.hpp
6. cpp/include/hsf/c_api.h
7. cpp/src/stat_engine.cpp (完整实现 10 维特征自适应提取、ThreadPool 多核分片并行，以及利用“总计数减去当前折计数”实现的 O(N) 高速 FitAndTransformOOF 折外特征提取)
8. cpp/src/c_api.cpp
9. cpp/apps/bench_and_test.cpp (含 RingBuffer 测试、步内翻转零泄露测试、OOF 翻转训练学生自身答案零自泄露断言测试，以及 1/2/4/8 线程多核吞吐压测)
```

### 【Step 2：交付 Python 配置、C++ 桥接与严谨数据流水线（第 2 批：5 个文件）】
```text
现在执行【第 2 批交付】：请严格按照定稿设计书第 2.2 节与第 3 章数据协议，输出以下 5 个文件的 100% 完整代码（严禁任何省略）：
1. requirements.txt
2. configs/default.yaml
3. python/config.py (@dataclass(frozen=True) 强类型配置与 YAML 加载)
4. python/cpp_bridge.py (零拷贝 ctypes 调用 C++ 动态库含 OOF 接口 + 公式 100% 对齐的 PurePythonStatEngine)
5. python/data_pipeline.py (完整支持 ASSISTments2009 / ASSISTments2017 CSV 解析并准确设置 has_real_timestamp / has_real_resp_time、仿真数据生成、仅基于 Train 构建词表并预留 <UNK>、按学生划分 70/10/20、训练集 5 折 OOF 先验特征构建，以及支持 window_aligned 与 full_history_carry 双模式的 KTDataset)
```

### 【Step 3：交付六大受控对照模型与严格选模评估器（第 3 批：2 个核心文件）】
```text
现在执行【第 3 批交付】：请严格按照定稿设计书第 5 章与第 6.1 节规格，输出以下 2 个文件的 100% 完整代码（每一步 Tensor 运算行尾必须加中文形状注释如 `# [B, L, D]`；确保 HSF_Concat 与 HSF_Gate 拥有完全相同的输入、编码器与题目先验对数赔率偏置）：
1. python/models.py (完整实现 StatOnlyLR, DKT, SAKT, CausalSeqOnly, HSF_Concat, HSF_Gate 六个模型及三组特征消融掩码逻辑)
2. python/trainer.py (严格仅使用 Val AUC 执行 EarlyStopping 选模，最后仅在最优权重上评测一次 Test 集；完整返回全局 AUC/ACC/RMSE、4 个学生历史步数区间 AUC 以及 4 个题目训练频次区间 AUC)
```

### 【Step 4：交付五大实验脚本、5 张论文绘图器、严谨单测套与 README（第 4 批收官：4 个文件）】
```text
现在执行最后一批【第 4 批交付】：请严格按照定稿设计书第 6.2~7 章规格，输出剩余 4 个文件的 100% 完整代码（确保运行测试集与主实验流水线开箱即用，诚实报告各项指标与加速比）：
1. python/run_experiments.py (支持单随机种子快速验证与多随机种子正式运行，自动完成主实验阶梯对比、短历史与低频题双分组分析、三类特征消融、分块视野归因对比、Python vs C++ OOF 多线程加速评测，导出 JSON、Markdown 表与 LaTeX 三线表)
2. python/plot_figures.py (绘制 fig1 至 fig5 共 5 张 300-DPI 论文高清图表并保存 .png 与 .pdf，含群体门控均值阴影曲线图)
3. tests/test_rigor.py (完整实现第 7 章规定的 4 项 pytest 严谨性验收测试)
4. README.md (准确诚实地说明架构设计、数据协议严谨性保障、Windows MinGW / Linux 编译命令、ASSISTments 真实数据集放置路径及完整实验复现步骤)
```
