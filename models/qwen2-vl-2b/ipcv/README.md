# Qwen2-VL-2B × IPCV

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/Perkzi/IPCV/tree/29726f044f0f9c0c2be4eb0005888ddeb5fa188e

许可记录：Apache-2.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_MATRIX_AVAILABLE_WITH_SEPARATE_QUALITY_TIMING_PINS

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前使用作者 Qwen2-VL 视觉分支及 repair R4 兼容修补，保留全局变化量 top-k、原位置 RoPE、全局 kNN、delta 重建与 AS 层序列。统一 4.57.6 eager 算术和保留 token 段首归属修补已在运行记录中激活；不走 PiToMe parent-mean 回填。当前 keep=0.85/0.70，LLM Sparse 关闭。原 Dense 严格失败仍成立，不能用静态规则对应替代数值归因。

QWEN_COMPLETED_MATRIX.json 汇总 360 行、10 个 job（含 Ours）；完整质量与 sampled280 计时是不同执行链。这里只发布范围说明，不发布原始质量/计时 JSON 或逐请求内容。

## Dense 严格门与公平性边界

Qwen 原严格门 strict_r6_parity_result=failed_preserved：同精度、同请求 ID 的 canonical Dense 与 IPCV repair R4 Dense 完整生成 token 序列未全部相同。FP16 的 MMMU_val 有 11 条、MMStar 有 1 条 token/text 差异；FP32 未观察到 token/text 差异。该门不是浮点 tensor/logit bitwise 测试。残余差异根因未定，不能称为已证无害的数值噪声。后续 non-parity 检查不把原严格失败改成通过。

质量分母来源：IPCV 使用其 repair R4 native Dense；R6 iLLaVA/DToMe/PiToMe 的 Dense quality 复用 canonical Dense，并保留导入来源记录。计时使用各自 sampled280 job 内实际测得的 Dense 配对。不能宣称五方法共享内部完全相同的 Dense，也不能声称所有质量 Dense 都由各方法独立重新生成。MMMU FP16 IPCV keep70 相对其 native Dense 的保留率为 99.6656%，不能换成 canonical 分母后称为 100%。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [IPCV加载及生成兼容接口](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/method_runtime.py)
- [FP16文本QK兼容](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/fp16_eager_stability.py)

本批提供项目侧接口快照，未带齐IPCV作者包、repair资源、冻结evaluator及完整运行依赖。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
