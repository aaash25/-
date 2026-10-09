# LLaVA-OneVision-7B × DToMe/DyMU

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/MikeWangWZHL/dymu/tree/3770d6add8fd9b972ae611f3f3b16a552dea13cc

许可记录：MIT (preserve per-component notices)。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：MISSING_CURRENT_RESULT_BINDING

审查分类：CURRENT_RESULT_BINDING_INCOMPLETE

本次已核材料中，尚缺与当前 PiToMe FINAL15 相同层级的方法专属结果绑定。此处不以旧历史结果替代当前版本，也不由材料缺口推断该方法从未运行。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [固定作者节点读取器](../../../migration_sources/current_archive/releases/onevision_formal_integration_20260930_r1/source/route/hypotheses/v615_h307_onevision_ipcv_global/author_visual_source.py)
- [共享视觉调用接口](../../../migration_sources/current_archive/releases/onevision_formal_integration_20260930_r1/source/route/hypotheses/v615_h307_onevision_ipcv_global/onevision_ports.py)

本批未带齐author_dtome_bridge与冻结作者资源；读取器/调用接口不等于作者DToMe算法文件已经收入。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
