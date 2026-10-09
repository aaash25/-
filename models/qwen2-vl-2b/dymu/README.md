# Qwen2-VL-2B × DToMe/DyMU

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/MikeWangWZHL/dymu/tree/3770d6add8fd9b972ae611f3f3b16a552dea13cc

许可记录：MIT (preserve per-component notices)。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_MATRIX_AVAILABLE_WITH_SEPARATE_QUALITY_TIMING_PINS

审查分类：FIXED_SCALAR_THRESHOLD_VARIANT

当前保留作者 DToMe 推理核的关键匹配、累计 size 加权及 log(size) 注意力规则，但以 0.985/0.9775/0.970 三组全层统一标量阈值替代完整方法的逐层统计校准/阈值向量，并从零基层 1 开始合并。应称“DToMe 推理核的固定标量阈值视觉迁移变体”，不能归成唯一必要的接口改动。FP32 评分/分组归约、目标坐标 RoPE 和 PatchMerger 前恢复另作数值/宿主政策披露；尚未量化这些差异对正式质量的影响。

QWEN_COMPLETED_MATRIX.json 汇总 360 行、10 个 job（含 Ours）；完整质量与 sampled280 计时是不同执行链。这里只发布范围说明，不发布原始质量/计时 JSON 或逐请求内容。

## Dense 严格门与公平性边界

Qwen 原严格门 strict_r6_parity_result=failed_preserved：同精度、同请求 ID 的 canonical Dense 与 IPCV repair R4 Dense 完整生成 token 序列未全部相同。FP16 的 MMMU_val 有 11 条、MMStar 有 1 条 token/text 差异；FP32 未观察到 token/text 差异。该门不是浮点 tensor/logit bitwise 测试。残余差异根因未定，不能称为已证无害的数值噪声。后续 non-parity 检查不把原严格失败改成通过。

质量分母来源：IPCV 使用其 repair R4 native Dense；R6 iLLaVA/DToMe/PiToMe 的 Dense quality 复用 canonical Dense，并保留导入来源记录。计时使用各自 sampled280 job 内实际测得的 Dense 配对。不能宣称五方法共享内部完全相同的 Dense，也不能声称所有质量 Dense 都由各方法独立重新生成。MMMU FP16 IPCV keep70 相对其 native Dense 的保留率为 99.6656%，不能换成 canonical 分母后称为 100%。

本仓库仅选入少量许可已标明的上游参考；方法部署代码、真实 prompts、数据和原始运行 journals 未在此公开。全局边界见 [审查状态说明](../../../sources/REVIEW_STATUS.md)。
