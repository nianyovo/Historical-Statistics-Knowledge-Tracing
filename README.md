# HSF-KT：融合历史统计特征的知识追踪

按 `docs/04_HSF-KT毕设完整工程设计书与GPT开发手册.md` 重建的研究工程。Python / PyTorch 负责数据处理、六模型受控实验与绘图；纯标准库 C++17 负责十维因果统计和学生级 OOF 特征提取。旧原型代码及二进制已移除，`docs/` 中的历史论文资料保留。

**已完成 ASSISTments2009 修正版上的 33 次正式训练、三种子汇总、五张论文图与实验章。** 主实验 Gate AUC 为 **0.7730 ± 0.0009**，Concat 为 **0.7716 ± 0.0011**，同骨干纯序列为 **0.7482 ± 0.0031**；均为固定学生划分下的均值 ± 样本标准差，不表示显著性结论。完整分析见 [真实数据实验结果与论文第五章](docs/08_真实数据实验结果与论文第五章.md)。本项目的 C++ 加速范围为统计特征构建。

## 1. 本机快速运行

在仓库根目录打开 PowerShell。本次已建立 `.venv`、编译 `build/libhsf_stat.dll`，直接运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m python.run_experiments --quick --engine cpp --output outputs/my-first-run
```

`--quick` 固定使用 60 名合成学生、32 步上下文、16 维隐藏层、2 个 epoch、1 个种子，执行所有实验和绘图。每次指定新的输出目录，程序拒绝覆盖已有实验记录。默认 Python 可能没有依赖，请使用 `.venv` 中的解释器。

重新建立环境（推荐 Python 3.11 / 3.12）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 给出兼容范围，`requirements-lock.txt` 保留早期 CPU 环境快照，`requirements-cuda-lock.txt` 记录正式实验的 CUDA 环境。本机现为 Python 3.12、PyTorch 2.14.1+cu130、RTX 3050 Ti Laptop GPU（4GB），`device: auto` 会检测可用设备。复现该 CUDA 环境可使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

## 2. 构建 C++17 引擎

Windows / MinGW（需要 `g++`、`mingw32-make`、CMake 位于 PATH）：

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 1
ctest --test-dir build --output-on-failure
.\build\bench_and_test.exe
```

本机 MinGW 并行 make 曾出现 `invalid output sync mutex`，因此以上使用单任务构建；**运行时特征引擎仍正常支持多线程**。编译选项为 `-O3 -Wall -Wextra -Wpedantic`，MinGW 动态库静态链接编译器运行库，减少 DLL 搜索路径问题。

Linux / GCC：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q
.venv/bin/python -m python.run_experiments --quick --engine cpp --output outputs/linux-smoke
```

Linux 命令和平台分支已提供；本次实际验证环境是 Windows / MinGW，PyTorch 的 CPU 与 CUDA 路径均已测试，Linux 尚未实测。`auto` 后端在未找到动态库时使用纯 Python；`cpp` 后端要求动态库存在；损坏的动态库会明确报错。可以通过 YAML 的 `experiment.cpp_library` 指定绝对路径。

## 3. 目录与入口

```text
CMakeLists.txt                 C++17 动态库、原生单测和吞吐压测
requirements.txt               Python 依赖范围
requirements-lock.txt          实测环境版本快照
configs/default.yaml           冻结配置类对应的统一配置
cpp/include/hsf/
  types.hpp                   数据结构、错误码和常量
  ring_buffer.hpp             固定容量泛型环形缓冲区
  thread_pool.hpp             RAII 标准库线程池
  stat_engine.hpp             因果状态机与批量/OOF 接口
  c_api.h                     ctypes 使用的 C ABI
cpp/src/stat_engine.cpp        十维特征、折外差分、并行提取
cpp/src/c_api.cpp              异常隔离与句柄生命周期
cpp/apps/bench_and_test.cpp    原生测试及百万交互压测
python/config.py              YAML 校验和 frozen dataclass
python/cpp_bridge.py          NumPy / ctypes 桥接与 Python 参考实现
python/data_pipeline.py       数据清洗、学生划分、词表、分块、OOF
python/models.py              六个对照模型与消融
python/trainer.py             Val 选模、一次 Test 推理、缓存分组指标
python/run_experiments.py     实验矩阵、速度评测、JSON/Markdown/LaTeX 导出
python/plot_figures.py        从缓存结果绘制五张 PNG/PDF 图
tests/test_rigor.py            防泄露、跨语言、数据协议、训练集成验收
pytest.ini                    测试临时目录保存在仓库内
docs/                         设计依据及原有论文资料
outputs/                      检查点、预测、审计信息、表格和图（忽略提交）
```

所有 Python 命令使用 `python -m python.<模块名>`，并在仓库根目录运行。

## 4. 数据准备与清洗

默认 `synthetic` 无需下载数据：固定 `synthetic_seed`，用题目难度、概念练习增长及时间遗忘生成 1200 名学生，默认长度 50–350，包含跨 200 步分块的学生。数据生成种子、学生切分种子和模型训练种子分开保存；多模型和多训练种子使用同一份数据及划分。

实验入口不会自动下载数据。本机已从官方链接下载修正版 ASSISTments2009，文件位于 `data/raw/skill_builder_data_corrected.csv`，来源、摘要及解码方式见 `data/assist2009_source.json`。数据和运行产物不纳入版本控制。另行复现时请按数据提供者的许可取得数据；使用其他数据集时复制 `configs/default.yaml` 为新配置，再修改 `data.dataset` 和路径：

| 数据集 | dataset | path | 时间含义 |
| --- | --- | --- | --- |
| ASSISTments2009 | `assist2009` | `data/raw/skill_builder_data_corrected.csv` | `order_id` 排序；概念间隔为练习步数 |
| ASSISTments2017 | `assist2017` | `data/raw/anonymized_full_release_competition_dataset.csv` | `startTime` 的真实时间差 |
| EdNet-KT1 | `ednet` | `data/raw/EdNet/KT1`（含 `u*.csv`） | `timestamp` 毫秒转秒 |

EdNet 还需设置 `questions_path: data/raw/EdNet/contents/questions.csv`，使用题目元数据中的答案计算 `correct`，使用 `tags` 作为知识点；`elapsed_time` 毫秒转秒。EdNet 全量很大，当前读取器会将数据装入内存，应先使用自己选定的固定学生子集评估内存需求。

2009 必须有 `user_id, problem_id, skill_id, correct, order_id`；若 `ms_first_response` 存在且有有效值，则毫秒转秒，否则 `has_real_resp_time=False`、特征 `[2]` 为零。2017 必须有 `studentId, problemId, skill`（或 `skill_id`）、`correct, startTime, timeTaken`；`timeTaken` 按秒读取，`startTime` 支持 Unix 秒、Unix 毫秒和日期文本。日期文本按 UTC 解析；数值绝对量级大于 `1e11` 时识别为毫秒。存在耗时数据时截断到 `[0.5, 600]`。

清洗移除缺失 ID/知识点、非二元标签、非法排序/时间值，随后稳定排序，再移除交互少于 5 次的学生。若耗时列有效但部分记录缺失，该部分记录被过滤，不用测试集均值填补。

**一次作答只能贡献一次标签。** 2009 同一 `(user_id, order_id)` 的多知识点重复行合并为排序后的复合概念；2017/EdNet 使用 `(学生, 题目, 时间戳)` 去重。重复行的题目、答案、时间或耗时冲突会报错。多知识点以复合概念作为单 ID，未实现多标签知识点注意力。这一约定固定用于全部模型。

数据字段依据：[ASSISTments2009 官方说明](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)、[ASSISTments2017 官方入口](https://sites.google.com/view/assistmentsdatamining/data-mining-competition-2017)、[EdNet 官方仓库](https://github.com/riiid/ednet)。真实发布版本的列名若不同，应先显式适配；程序不会猜测缺失列。

## 5. 实验防泄露与公平性约定

1. 按学生固定划分 70% Train / 10% Val / 20% Test；同一学生不会跨集合。词表只由 Train 建立：Padding=0，已见 ID 从 1 开始，Embedding 的 UNK 为训练词表大小+1。统计引擎为未见题目/概念保留独立的负 ID，使先验回退到训练均值，同时不会把不同新概念的个人历史混在一起。
2. Train 内部按学生分 5 折，同一学生的所有块在同一折。C++ 和 Python 都用总统计减本折统计，在该学生的所有交互上排除其所在折的标签和耗时；Val/Test 只使用完整 Train 的先验。独立测试还以逐折重拟合验证差分算法。
3. 所有十维特征先提取，再用当前答案更新个人状态；所有序列模型统一右移历史交互一位。当前题目和概念已知，当前答案不作为预测输入。模型返回 logits，训练使用 `BCEWithLogitsLoss`。
4. 主实验 `window_aligned` 先分块再统计，默认每块 200 步，序列骨干和统计分支的个人历史视野相同。附加 `full_history_carry` 在整条学生历史上统计后分块。两种模式保留完全相同的样本与原始步号，长学生指标另外报告后续块上的 AUC。
5. `HSF_Concat` / `HSF_Gate` 使用相同的嵌入、Transformer、统计编码器、预测头、可学习先验偏置、融合参数尺寸和初始化。三组消融分别屏蔽 `[0:3]`、`[3:7]`、`[7:10]`；移除先验组时同步关闭先验偏置。
6. 每个 epoch 只评估 Val，按 Val AUC 保存最佳权重。训练结束加载该权重，仅进行一次 Test 前向推理。分组评估、视野分析和门控曲线全部复用这次预测缓存。每个独立的预先定义模型/种子/消融运行各自有一次最终测试，不能据此继续在同一测试集上调参。
7. 冷启动分组使用**学生真实原始步号**，不会把每个块的前五题都算作新学生；题目频次来自完整 Train。空组或单一标签组 AUC 为 `null` / `N/A`，同时导出样本数与正例数。缺失指标不会补成 0.5。

`StatOnlyLR` 遵循设计书，输入 10 维统计与两个先验 logit，经过 `LayerNorm(12) -> Linear(12,1)`；它是该归一化特征空间上的逻辑回归，不等同于原始特征上无变换的传统 LR。DKT（GRU）和 SAKT 为接口统一后的参考实现，包含当前题目/概念表征，不是原论文所有数据处理与超参数的逐项复刻。不能将这里的指标直接等同于原论文榜单。

## 6. 正式实验与输出

```powershell
# 默认：1200 名合成学生、三个训练种子、最多 30 epoch；不是实际数据论文实验
.\.venv\Scripts\python.exe -m python.run_experiments --engine cpp --output outputs/synthetic-full

# 使用自己已准备好的真实数据配置
.\.venv\Scripts\python.exe -m python.run_experiments --config configs/assist2009.yaml --engine cpp --output outputs/assist2009-final

# 仅在原进程已退出后恢复：核对配置、源码、数据摘要与 PyTorch 版本
.\.venv\Scripts\python.exe -m python.run_experiments --config configs/assist2009.yaml --engine cpp --output outputs/assist2009-final --resume

# 仅六模型主实验；可从保存的预测计算其他分组，不重跑 Test
.\.venv\Scripts\python.exe -m python.run_experiments --suite main --output outputs/main-only

# 仅统计引擎压测（含当前训练集规模及 50K / 200K / 1M）
.\.venv\Scripts\python.exe -m python.run_experiments --suite benchmark --engine cpp --output outputs/speed-only

# 三种子快速流程验证
.\.venv\Scripts\python.exe -m python.run_experiments --quick --seeds 42 2025 2026 --output outputs/quick-three-seeds

# 只从缓存结果重绘图片，不访问模型或重复测试推理
.\.venv\Scripts\python.exe -m python.plot_figures outputs/quick-three-seeds
```

完整实验运行六个主模型、Gate 的三类消融，以及 Concat/Gate 两个 carry 变体，每个种子共 11 个训练任务。`chunk_protocol` 在单独调用 `prepare_data` 时可选择两种模式；受控实验总入口要求默认 `window_aligned`，自动在附加阶段运行 carry，防止主表混入不同视野。

输出包括：

- `manifest.json` / `config.yaml`：有效配置、运行环境、源码摘要、计时范围、是否 quick。
- `data_audit.json`：实际数据量、元数据标记、学生划分、折归属、训练词表及题目频次。
- `runs/<实验名>/best.pt` / `last.pt` / `history.json` / `result.json` / `test_predictions.npz`：最佳权重、含优化器与随机数状态的断点、训练/验证曲线、一次测试结果及预测缓存。
- `results.json` / `summary.json` / `benchmarks.json`：逐种子指标、聚合指标与耗时原始统计。
- `tables.md` / `tables.tex`：主表、Markdown 分组/长历史/速度表和可用 `booktabs` 排版的 LaTeX 主表。
- `figures/fig1_...` 至 `fig5_...`：五张 300-DPI PNG 及矢量 PDF。只运行部分实验时仅生成有数据支撑的图片。

多种子结果为均值 ± **样本标准差**，不是置信区间；单种子 SD=0 不代表稳定。门控图先对同一学生同一步的不同种子求均值，再计算学生之间的均值和 SD；案例按长度和学生编号确定，不挑选最符合预期的曲线。

基准计时先预热，再测 OOF 拟合+提取，包含输出分配、输入检查和线程启动，不含 CSV 读取与预打包。两种语言使用相同的计数差分算法；扩大规模时循环/截断现有序列并保留原学生的折归属，仅用于计算性能，不用于模型学习。结果必须满足跨语言最大绝对误差 `<1e-6`。短数据上线程启动开销可能抵消收益，不承诺线程数越多越快。

## 7. C ABI 和复杂度说明

`PackedBatch` 使用每行 32 字节、字段布局与 C 结构体对齐的结构化 NumPy 数组：四个 int32 + 两个 float64；offsets 为 int64。打包发生一次，C 调用直接读入 NumPy 连续内存、写入 float32 输出，不在 ABI 内部复制交互数组。“零拷贝”仅指此调用边界，不表示 CSV、数据清洗、分块或张量构造完全无拷贝。

五个 C 导出函数均通过异常隔离返回状态码：0 成功，-1 输入错误，-2 分配失败，-3 其他异常。句柄不能跨调用并发使用，调用者需保证原始指针有效。Python 桥接负责生命周期、折一致性检查及关闭后拒绝调用。

环形窗口使用固定栈存储，运行时窗口范围 1–1024，滑动和更新为 O(1)。按学生的概念状态使用哈希表，存储量随该学生见过的概念数增长；不是整个状态严格 O(1) 空间。全局题目/概念计数对紧凑正数 ID 使用连续数组，负数及超大稀疏 ID 使用哈希回退；OOF 查询使用总计数减本折计数。固定五折下总体期望时间为 O(N)，聚合空间随 ID 数量和折数增长。普通提取由编译期模板去掉排除折查询，线程按连续学生区间分片。原生压测不计 Python、打包或神经网络耗时，勿与跨语言评测混用。

真实数据路径使用列数组和向量化编码，学生分块复用连续数组视图；旧对象输入接口保留用于参考测试。优化前后的真实数据特征和全部 Dataset 张量逐项一致。评估时裁剪右侧 Padding，训练保持原样，避免改变 Dropout 随机数分配。按长度重新组 Batch 和 AMP 均未启用，因为它们不能保证既定训练结果完全不变。完整耗时、内存和数值证据见 [性能优化与等价性验收](docs/06_性能优化与等价性验收.md)。

## 8. 验收与使用边界

`pytest` 验证 OOF 自泄露、未来答案扰动、原生/参考数值一致、块边界重置、未知概念独立性、初始化公平、消融偏置关闭、Padding 前反向传播、真实数据列/单位/重复作答，以及 Test 仅在训练结束后调用一次。`ctest` 另行验证 C++ 原生行为。临时测试文件位于仓库 `.pytest_tmp`。

当前通过 63 项 pytest 与 C++ 原生测试。正式实验在 2026-11-13 完成，输出位于 `outputs/assist2009-final/`；33 次运行均保存一次最终 Test 预测，重新计算全部缓存指标与原结果完全一致，源码与数据摘要一致。正式协议见 [docs/07](docs/07_ASSISTments2009正式实验协议.md)，结果与局限见 [docs/08](docs/08_真实数据实验结果与论文第五章.md)。`docs/00`–`03` 包含早期写作材料，其旧指标、旧目录和性能口号不作为本版实测证据。

本次关键边界：Gate 相对 Concat 的整体提升约 0.144 个 AUC 百分点，在未见题目组反而略低；低频题增益较大，但“学生历史越短增益越大”未获支持。近期状态消融几乎不影响整体指标，不能声称每组统计均带来强收益。当前仅有一个真实数据集、一个学生切分，DKT/SAKT 是适配实现。

性能优化独立批次中，数据准备中位耗时从 6.209 秒降至 2.910 秒；百万交互四线程 OOF 从旧 C++ 的 81.839 ms 降至 46.174 ms。正式实验后的跨语言批次另测得百万四线程 103.331 ms，对 Python 参考为 173.05 倍。不同批次的系统运行状态与测量对象不同，应保留各自原始记录，不用较小耗时替换其他批次结果，也不与神经网络训练速度混淆。

运行目录还包含 `analysis_audit.json`（缓存重算与配对差值）、`source_snapshot.zip`（冻结源码）、`snapshot_hashes.json`（源码与 DLL 摘要）和 `hardware_notes.json`。图表为 5 张 PNG 与对应矢量 PDF。数据、检查点及实验产物均被 `.gitignore` 忽略，迁移或提交前需单独备份所需证据。跨机器重跑应使用新的输出目录；缓存 JSON 中保存的是本机绝对路径，迁移旧产物后重绘须先在副本中更新这些路径。
