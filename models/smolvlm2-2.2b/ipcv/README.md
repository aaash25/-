# SmolVLM2-2.2B × IPCV

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/Perkzi/IPCV/tree/29726f044f0f9c0c2be4eb0005888ddeb5fa188e

许可记录：Apache-2.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_CORRECTED_MATRIX_AVAILABLE

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前修复版 port 96d046d356ff100a6eaf4040284b450f89965b54f116aefc45fb2238de5e4d5c 参照作者 InternVL2 视觉分支：逐 crop 选择，全 encoder batch 的 kNN/delta 重建，AS 时序及最终原位恢复。旧逐 crop 邻域缺陷不归到该版本。当前质量和 R4 runner 逐请求执行；该端口不保证独立请求合批隔离。官方 4.57.6 宿主快照尚未匹配实际 installed 模块 SHA，不能据版本字符串宣称完整部署字节已核实。

SMOL_COMPLETED_QUALITY_R4_TIMING.json 的 180 行仅覆盖 corrected IPCV/DToMe，不用于推断其它方法质量缺失。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [Smol IPCV迁移实现](../../../migration_sources/current_archive/route/releases/smol_existing_configuration_correction_20261001_r1/source/route/benchmarks/smolvlm_full_reproduction_20260903/migration/smol_ipcv_port.py)
- [共享方法调度](../../../migration_sources/current_archive/route/releases/smol_existing_configuration_correction_20261001_r1/source/route/benchmarks/smolvlm_full_reproduction_20260903/production/peer_common.py)

还依赖原目录中的smol_merge_ports、冻结资源及环境；参考分支的InternVL MIT与IPCV新增部分Apache许可分别保留。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
