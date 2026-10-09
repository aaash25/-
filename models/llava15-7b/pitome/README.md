# LLaVA1.5-7B × PiToMe

本目录是迁移状态说明，不包含可运行的模型适配器。完整科学验收 NOT_ASSERTED；下述静态证据不等于全栈/GPU数值等价或全部样本复跑。质量、计时、源码身份分别绑定。验收必须遵循冻结判据，明确保留严格失败和已证方法差异，不能以“非 bitwise 通常可接受”覆盖已经失败的门。

作者来源：https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40

许可记录：CC-BY-NC-4.0。第三方许可仅按选入文件适用；不为整个仓库重新授权。
## 当前材料与有限结论

材料状态：CURRENT_FP32_R096_FINAL15_AVAILABLE

审查分类：LIMITED_CORE_STATIC_SUPPORT

当前 FP32 r=0.96 FINAL15 摘要为 15 数据集、32,338 配对质量请求、280 计时条目及 15 份原始 exit receipt。结果摘要不替代完整执行证据；该有限材料状态不等于科学全验收或跨宿主等价。

原始题干、数据和运行journals不随源码快照公开。上文保留各组合的科学证据边界；源码发布不把未通过或未执行的验收改为通过。

## 当前公开源码入口

- [冻结PiToMe载荷及原门控脚本](../../../migration_sources/current_archive/releases/llava_r7_fidelity_r5_20261001_r1/route/hypotheses/v611_h303_llava15_five_method_fp32_a800/frozen_sources/pitome_author_fp16.py)
- [LLaVA运行接口](../../../migration_sources/current_archive/releases/llava_r7_fidelity_r5_20261001_r1/route/hypotheses/v611_h303_llava15_five_method_fp32_a800/llava_runtime.py)
- [带归属说明的解码HF-CLIP参考](../../../migration_sources/decoded_author_payloads/pitome/patch/clip_hf.py)

原门控脚本名字中的FP16不替代本页当前FP32结果身份；原载荷与新增归属说明的明文副本hash分开记录。仍缺完整冻结桥接/环境。

这些是文件快照、选集或调用接口，不是完整可复跑发行包。逐文件取舍见 [FILE_DISPOSITIONS](../../../sources/FILE_DISPOSITIONS.json)；具体阻断见 [运行限制](../../../sources/MIGRATION_RUNTIME_LIMITS.md)；许可见 [组件说明](../../../third_party/COMPONENT_NOTICES.md)。返回 [仓库导航](../../../README.md)。
