# SmolVLM2-2.2B × PiToMe

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40

许可记录：CC-BY-NC-4.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：REUSED_QUALITY_BINDINGS_AVAILABLE_VERIFIED12

审查分类：LIMITED_CORE_STATIC_SUPPORT

已有质量 .93/.94/.95 与 H236 .97、R4 计时绑定到同一 d550750bb8a191b25bad46161f9ae3fef1131fe4ec32ed6e29601d07096cfe61 合并核心；不能因 corrected180 不含 PiToMe 就标为质量不存在。跨方法联合可比仍限 VERIFIED12；POPE、MMVP、MME 的旧 selector 未统一。静态核对支持作者 HF-CLIP 的 BSM/energy、普通 mean 与层后合并规则。无 CLS、27 层以及恢复 729 slot 后经过原 post-LN/scale-3 connector 是宿主迁移政策，不是作者 compact 下游的端到端等价证明。

SMOL_COMPLETED_QUALITY_R4_TIMING.json 的 180 行仅覆盖 corrected IPCV/DToMe，不用于推断其它方法质量缺失。

本仓库仅选入少量许可已标明的上游参考；方法部署代码、真实 prompts、数据和原始运行 journals 未在此公开。全局边界见 [审查状态说明](../../../sources/REVIEW_STATUS.md)。
