# SmolVLM2-2.2B × iLLaVA

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hulianyuyy/iLLaVA/tree/d2d291877ba9c7d4781cdc9ed1820661c8e47291

许可记录：METHOD_ROOT_LICENSE_NOT_FOUND_AT_PINNED_COMMIT。第三方许可仅按选入文件适用；不为整个仓库重新授权。
上游固定 commit 未发现方法根级 LICENSE；独立说明和上游链接不等于源码再分发授权。本目录不收入 iLLaVA 代码。

## 当前材料与有限结论

材料状态：REUSED_QUALITY_SOURCE_MAPPING_AVAILABLE_VERIFIED12

审查分类：LIMITED_CORE_STATIC_SUPPORT

现有质量来源映射与 R4 计时配对覆盖 remove64/104/136/152/160 五档：每精度 75 个数据集点，FP16/FP32 合计 150 点。其中联合可比 VERIFIED12 为每精度 60 点；POPE、MMVP、MME 的旧 selector 尚未统一，不能声称 15 集统一验收通过。该来源映射不等于本次重新读取全部原始质量输出。静态核对支持所选作者 SigLIP 自选择分支的评分、选择、排名权重、插入时序和数量规则；FP32 不再强制 half 舍入，出口恢复到 729 slot 以保留宿主 connector，属于应披露的扩展/迁移。

SMOL_COMPLETED_QUALITY_R4_TIMING.json 的 180 行仅覆盖 corrected IPCV/DToMe，不用于推断其它方法质量缺失。

本仓库仅选入少量许可已标明的上游参考；方法部署代码、真实 prompts、数据和原始运行 journals 未在此公开。全局边界见 [审查状态说明](../../../sources/REVIEW_STATUS.md)。
