# VLM 视觉编码器方法迁移

本仓库维护四个VLM模型、四种方法的迁移说明、源码快照与独立核对。目前已有16个模型×方法说明目录，以及包含33个Python文件的公开源码快照，涵盖完整项目文件、明确标记的节点选集和上游参考副本。

**当前不是16组完整可运行的发行包，也不是16组已通过复现验收的声明。** 不同组合公开到实现、选集或调用接口的不同层级；每个目录单独注明。iLLaVA作者算法尚未随公开包收入。

## 从这里开始

- [源码快照说明](SOURCE_CANDIDATE_README.md)：本批实际内容、保留身份和公开范围
- [完整项目文件快照](migration_sources/current_archive/) · [节点选集](migration_sources/extracted_components/) · [解码作者参考](migration_sources/decoded_author_payloads/) · [上游参考副本](migration_sources/upstream_references/)
- [运行限制与缺失依赖](sources/MIGRATION_RUNTIME_LIMITS.md)：先读阻断，再考虑后续封装
- [模型×方法索引](sources/MODEL_METHOD_INDEX.json) · [逐文件取舍与来源](sources/FILE_DISPOSITIONS.json)
- [组件许可与版权](third_party/COMPONENT_NOTICES.md) · [HF/InternVL补充出处](sources/ADDITIONAL_COMPONENT_NOTICE_PROVENANCE.json)

## 模型与方法入口

| 模型 | iLLaVA | PiToMe | DToMe/DyMU | IPCV |
|---|---|---|---|---|
| Qwen2-VL-2B | [说明与调用入口](models/qwen2-vl-2b/illava/README.md) | [说明与源码](models/qwen2-vl-2b/pitome/README.md) | [说明与源码](models/qwen2-vl-2b/dymu/README.md) | [说明与接口](models/qwen2-vl-2b/ipcv/README.md) |
| SmolVLM2-2.2B | [说明与调用入口](models/smolvlm2-2.2b/illava/README.md) | [说明与选集](models/smolvlm2-2.2b/pitome/README.md) | [说明与源码](models/smolvlm2-2.2b/dymu/README.md) | [说明与源码](models/smolvlm2-2.2b/ipcv/README.md) |
| LLaVA-1.5-7B | [说明与调用入口](models/llava15-7b/illava/README.md) | [说明与源码](models/llava15-7b/pitome/README.md) | [说明与接口](models/llava15-7b/dymu/README.md) | [说明与接口](models/llava15-7b/ipcv/README.md) |
| LLaVA-OneVision-7B | [说明与加载入口](models/llava-onevision-7b/illava/README.md) | [说明与源码](models/llava-onevision-7b/pitome/README.md) | [说明与加载入口](models/llava-onevision-7b/dymu/README.md) | [说明与接口](models/llava-onevision-7b/ipcv/README.md) |

实际checkpoint ID、revision、dtype与后端以对应冻结版本为准，不能只凭模型简称混用。研究范围是视觉编码器分支，不把LLM token减少带来的收益算作ViT加速比。

## 运行与验收边界

- Qwen原适配顶层仍导入选集中省略的FiCoCo函数，单选PiToMe/DToMe也存在导入阻断。
- Smol选集仍保留指向省略iLLaVA函数的分支；OneVision和LLaVA还缺若干冻结桥接、资源与环境文件。
- 全文/逐节点AST一致支持清理或保留节点未改，不证明选集具有原模块全部功能。不要把选集hash替换进旧冻结合同以绕过缺口。
- Qwen Dense原严格门为failed_preserved，比较完整生成token序列；残余差异根因未定。它不因代码公开或静态核对而改为通过。
- Qwen DToMe固定标量阈值变体、Smol DToMe dispatch/none-pass差异，以及其它迁移边界均在各组合说明中保留。

质量与计时分别绑定各自配置、源码和执行来源；后续封装或修复不得追溯改写旧结果身份。main表示开发状态，不自动表示已验证版本；当前未发布全组合验收通过的release。

## 许可与公开范围

许可按文件及来源组件保留：PiToMe的CC BY-NC 4.0非商业条件、DyMU MIT、IPCV/HF/LLaVA相关Apache-2.0、InternVL底层MIT分别处理，不为全仓统一重新授权。iLLaVA方法根许可未明确，其作者实现未收录；公开加载器和调用配置不替代授权。

模型权重、数据集实体、真实基准题干、私人会话、凭据、个人机器路径和未筛选原始日志不进入仓库。通用对话模板仍按源码保留。

## 文件身份与历史清单

- [当前统一公开清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261009_r3.json)：覆盖本次导航更新后的全部文件，清单自身除外
- [源码快照静态复核](sources/SOURCE_REVIEW_VERIFICATION_20261009_r2.json)：记录源码身份和检查范围，不是运行验收
- 根MANIFEST.json保留为[首批29文件的历史清单](https://github.com/aaash25/-/blob/ed7b06b65c5e563106671e76fe37ce4e0f2cac78/MANIFEST.json)
- r2统一清单保留为[第二批公开后的历史快照](https://github.com/aaash25/-/blob/e417f21c1dde59bd6962a07cacbe628adf761332/sources/PUBLIC_REPOSITORY_MANIFEST_20261009_r2.json)

历史清单的hash对应链接中的固定commit，不要求它们匹配随后更新的文档。最新清单记录历史清单文件本身，历史清单不反向引用最新清单，避免相互哈希循环。
