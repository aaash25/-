# LLaVA-OneVision-7B × PiToMe

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40

许可记录：CC-BY-NC-4.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_FP32_R096_FINAL15_AVAILABLE

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前 FP32 r=0.96 FINAL15 摘要为 15 数据集、32,338 配对质量请求、280 计时条目；14 份原始 exit receipt，MMVP receipt 仍缺失。冻结宿主为 26 层、每 crop 729 slot，逐层压缩最终 260，再恢复 729；729 的网格边长 27 不是视觉层数。静态核对支持所选作者 HF-CLIP 无 CLS 核心规则。crop-local 计算、原 slot 父映射、出口恢复及局部确定性归约需披露；不能把摘要/源码身份提升为所有生成进程的 import 链已重核或端到端等价。

本仓库仅选入少量许可已标明的上游参考；方法部署代码、真实 prompts、数据和原始运行 journals 未在此公开。全局边界见 [审查状态说明](../../../sources/REVIEW_STATUS.md)。
