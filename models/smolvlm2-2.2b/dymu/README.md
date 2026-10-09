# SmolVLM2-2.2B × DToMe/DyMU

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/MikeWangWZHL/dymu/tree/3770d6add8fd9b972ae611f3f3b16a552dea13cc

许可记录：MIT (preserve per-component notices)。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_CORRECTED_MATRIX_AVAILABLE

审查分类：CONFIRMED_AUTHOR_RULE_NON_EQUIVALENCE

已确认当前完整 dispatch 与作者固定 SigLIP 推理路径非等价：当前保留 H125 batch 式 _merge，不按视觉 B=1 切换作者 instance helper；即便 B>1，none-pass 提前返回也与作者后续 compact/size 除法不同。不能仅以作者允许 batch 分支或首层 r=0 将反例排除。现有结果是冻结 adapter 的观测；正式数据实际触发范围、最终答案/准确率影响尚未量化，不得宣布全部成绩污染，也不得继续声称完整作者路径严格等价。

SMOL_COMPLETED_QUALITY_R4_TIMING.json 的 180 行仅覆盖 corrected IPCV/DToMe，不用于推断其它方法质量缺失。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [DToMe旧路径桥接](../../../migration_sources/current_archive/route/releases/smol_existing_configuration_correction_20261001_r1/source/route/benchmarks/smolvlm_full_reproduction_20260903/migration/dtome_legacy_bridge.py)
- [H125实现快照](../../../migration_sources/current_archive/route/releases/smol_existing_configuration_correction_20261001_r1/source/route/hypotheses/v439_h125_dtome_smolvlm_pope_full300/evaluate.py)
- [共享方法调度](../../../migration_sources/current_archive/route/releases/smol_existing_configuration_correction_20261001_r1/source/route/benchmarks/smolvlm_full_reproduction_20260903/production/peer_common.py)

作者阈值/源码资源和模型数据未带齐；保留已披露dispatch/none-pass差异，不以新封装替换旧结果身份。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
