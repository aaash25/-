# Qwen2-VL-2B × iLLaVA

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hulianyuyy/iLLaVA/tree/d2d291877ba9c7d4781cdc9ed1820661c8e47291

许可记录：METHOD_ROOT_LICENSE_NOT_FOUND_AT_PINNED_COMMIT。第三方许可仅按选入文件适用；不为整个仓库重新授权。
上游固定 commit 未发现方法根级 LICENSE；独立说明和上游链接不等于源码再分发授权。本目录不收入 iLLaVA 代码。

## 当前材料与有限结论

材料状态：CURRENT_MATRIX_AVAILABLE_WITH_SEPARATE_QUALITY_TIMING_PINS

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前 R6 采用固定作者 Qwen2-VL 视觉选择/合并核心，保留逐图预算、注意力重要性、低分 top-k、排名权重与 raw RoPE 同步。当前每层移除率为 0.06/0.15，关闭 LLM 合并；统一 eager 与 FP32 省略强制 half 舍入属于需披露的扩展。此路径直接以压缩序列进入 PatchMerger，不采用 parent slot restoration。

QWEN_COMPLETED_MATRIX.json 汇总 360 行、10 个 job（含 Ours）；完整质量与 sampled280 计时是不同执行链。这里只发布范围说明，不发布原始质量/计时 JSON 或逐请求内容。

## Dense 严格门与公平性边界

Qwen 原严格门 strict_r6_parity_result=failed_preserved：同精度、同请求 ID 的 canonical Dense 与 IPCV repair R4 Dense 完整生成 token 序列未全部相同。FP16 的 MMMU_val 有 11 条、MMStar 有 1 条 token/text 差异；FP32 未观察到 token/text 差异。该门不是浮点 tensor/logit bitwise 测试。残余差异根因未定，不能称为已证无害的数值噪声。后续 non-parity 检查不把原严格失败改成通过。

质量分母来源：IPCV 使用其 repair R4 native Dense；R6 iLLaVA/DToMe/PiToMe 的 Dense quality 复用 canonical Dense，并保留导入来源记录。计时使用各自 sampled280 job 内实际测得的 Dense 配对。不能宣称五方法共享内部完全相同的 Dense，也不能声称所有质量 Dense 都由各方法独立重新生成。MMMU FP16 IPCV keep70 相对其 native Dense 的保留率为 99.6656%，不能换成 canonical 分母后称为 100%。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [共享运行接口中的iLLaVA调用](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/method_runtime.py)
- [参数配置](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/grid.py)

iLLaVA作者实现及三份对应Qwen源码未随公开快照收入；共享调用/配置不等于iLLaVA核心已发布。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
