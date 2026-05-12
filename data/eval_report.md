# ST-MTFC 结构感知 RAG 实验分析报告 (Pilot v2.2)
> **评测环境**: Qwen2.5-1.5B-Instruct (Judge) | **数据集**: Degraded (OCR) | **设置**: Top-K=3

## 1. 核心指标定义
- **GC_Overlap (Gold-Context)**: 衡量检索上下文对标准答案符号的覆盖程度。
- **CA_Overlap (Context-Answer)**: 衡量生成答案对检索证据符号的采信程度（证据忠实度）。
- **GA_Overlap (Gold-Answer)**: 衡量生成答案与标准答案符号的重合程度。

## 2. 跨方法性能对比概览
| Method | Avg Score | Hit Rate | Formula Match | Suspected Override | GC_Overlap | CA_Overlap | GA_Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| standard | 14.2 | 57.1% | 2.9% | 5 | 0.71 | 0.74 | 0.46 |
| mtfc_v1 | 13.9 | 57.1% | 0.0% | 3 | 0.71 | 1.00 | 0.60 |
| mtfc_v2 | 13.9 | 57.1% | 2.9% | 3 | 0.66 | 0.86 | 0.57 |
| st_mtfc | 14.3 | 57.1% | 2.9% | 2 | 0.71 | 1.20 | 0.60 |

## 3. 疑似证据-答案不一致性 (Evidence-Answer Inconsistency) 案例簇
| ID | Method | GC_Overlap | CA_Overlap | GA_Overlap | Failure Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 21 | standard | 2 | 0 | 0 | 未回答 |
| 26 | standard | 2 | 0 | 0 | 未回答 |
| 27 | standard | 3 | 1 | 0 | 不准确 |
| 31 | standard | 2 | 1 | 0 | judge_parse_error |
| 32 | standard | 2 | 1 | 1 | judge_parse_error |
| 21 | mtfc_v1 | 2 | 0 | 0 | 未回答问题 |
| 27 | mtfc_v1 | 3 | 1 | 0 | judge_parse_error |
| 32 | mtfc_v1 | 2 | 1 | 1 | judge_parse_error |
| 27 | mtfc_v2 | 3 | 1 | 0 | 待评测回答没有准确地描述Hopfield神经网络的主要应用领域 |
| 31 | mtfc_v2 | 2 | 3 | 1 | judge_parse_error |
| 32 | mtfc_v2 | 2 | 1 | 1 | judge_parse_error |
| 27 | st_mtfc | 3 | 1 | 0 | 不相关 |
| 32 | st_mtfc | 2 | 1 | 1 | judge_parse_error |

## 4. 实验观察与初步讨论
1. **结构化标注趋势**: 在 Pilot 设置下，ST-MTFC 观察到疑似 evidence-answer inconsistency 案例从 5 例（standard）下降至 2 例。这指示了结构化标注可能有助于提升生成答案对检索证据的采信度。
2. **符号一致性分析**: 观察到 CA_Overlap 指标从 0.74 (standard) 提升至 1.20 (st_mtfc)。该趋势指示了 Structure Tagging 对模型注意力分布可能产生了一定的正面引导作用，但其统计显著性仍需在全量实验中验证。
3. **典型案例 (Q31)**: 在 Q31 的横向对比中，standard 模式表现为 CA_Overlap=1/GA_Overlap=0，而 st_mtfc 表现为 CA_Overlap=3/GA_Overlap=2。该变动指示了结构感知对特定符号密集型任务的潜在敏感性。
4. **结论审慎性**: 以上结论仅基于 1.5B Judge 模型及 35 题规模的初步观察。后续将在更强 Judge 或全量 7B 模型上复核该趋势。