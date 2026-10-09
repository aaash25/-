# Qwen2-VL-2B × PiToMe

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40

许可记录：CC-BY-NC-4.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_MATRIX_AVAILABLE_WITH_SEPARATE_QUALITY_TIMING_PINS

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前 R6 core da30c93c4e39361df9a6f424af9cf2bcb3467c0b7fbd964cc1dbe8ff0d8e015a 使用独立 segment_reduce；历史 ee477dae 前缀归约缺陷不归到当前 R6。现有静态和数学检查支持 parent 方向、输出重排、代表索引同步及原 slot 顺序恢复；目标坐标 RoPE 与 restored-grid PatchMerger 是明确的宿主迁移政策，恢复不是原特征的无损逆运算。忠于所选 HF-CLIP 普通 mean 分支，不等于论文 size-weighted 伪代码逐式复现。

QWEN_COMPLETED_MATRIX.json 汇总 360 行、10 个 job（含 Ours）；完整质量与 sampled280 计时是不同执行链。这里只发布范围说明，不发布原始质量/计时 JSON 或逐请求内容。

## Dense 严格门与公平性边界

Qwen 原严格门 strict_r6_parity_result=failed_preserved：同精度、同请求 ID 的 canonical Dense 与 IPCV repair R4 Dense 完整生成 token 序列未全部相同。FP16 的 MMMU_val 有 11 条、MMStar 有 1 条 token/text 差异；FP32 未观察到 token/text 差异。该门不是浮点 tensor/logit bitwise 测试。残余差异根因未定，不能称为已证无害的数值噪声。后续 non-parity 检查不把原严格失败改成通过。

质量分母来源：IPCV 使用其 repair R4 native Dense；R6 iLLaVA/DToMe/PiToMe 的 Dense quality 复用 canonical Dense，并保留导入来源记录。计时使用各自 sampled280 job 内实际测得的 Dense 配对。不能宣称五方法共享内部完全相同的 Dense，也不能声称所有质量 Dense 都由各方法独立重新生成。MMMU FP16 IPCV keep70 相对其 native Dense 的保留率为 99.6656%，不能换成 canonical 分母后称为 100%。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [视觉宿主适配](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/qwen_peer_ports.py)
- [PiToMe/DToMe核心节点选集](../../../migration_sources/extracted_components/qwen_r6/peer_port_core.py)
- [运行接口](../../../migration_sources/current_archive/releases/qwen_minimal_peer_reuse_r6_20261001/source/method_runtime.py)

选集省略FiCoCo函数，但原视觉适配仍顶层导入它们；即使只选PiToMe也不是可直接运行组合。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
