# 实际迁移源码：第二批公开快照

本批是经逐文件检查的源码快照与明确标记的节点选集，供阅读、溯源和后续封装。它不是16组可直接运行的完整发行包，也不授予科学验收通过状态。没有修改评分、匹配、合并或恢复逻辑来使打包自洽。

## 运行限制，先读这一节

- Qwen：原 qwen_peer_ports.py 顶层仍导入 ficoco_compress/restore_ficoco，PiToMe/DToMe选集已省略这两个函数。即使把选集放回预期位置且只选择PiToMe/DToMe，也会遇到导入阻断；不能把此选集直接替换为完整冻结模块。
- Smol：PiToMe选集保留原 SmolMergePort 类，其中 illava_vit 分支仍引用省略的 illava_step。该分支不是可运行的iLLaVA实现，也不以伪造占位函数掩盖缺口。
- OneVision：onevision_ports.py 顶层引用未收入本批的 author_dtome_bridge/author_illava_bridge，因此单选PiToMe也不等于可独立导入运行。
- LLaVA及其它运行器还缺冻结桥接、算子、环境锁、数据/资产与完整目录布局。仅安装Torch/Transformers不能补齐本包。

准确文件、行号、外部资源与边界见 [运行依赖说明](sources/MIGRATION_RUNTIME_LIMITS.md)。后续独立封装必须另标版本并另审；不得把本批快照的原hash或旧结果身份移给改装后代码。

## 本批内容与来源

25份完整项目源码在加入注释/换行规范化后，与另存原始源的全文AST相同。另有3份选集，保留的25个顶层类/函数节点及其它顶层语句AST一致，但模块整体已省略不随本批发布的算法函数；两份Smol选集内容相同。还含2份解码PiToMe参考副本和3份上游参考源码，共33个Python文件。算法原载荷只做标准库解码/哈希检查，没有执行。

FILE_DISPOSITIONS逐项列原31文件。Qwen三份iLLaVA源码、含其合并函数的两份完整Smol模块以及含FiCoCo算法的完整Qwen core不在本批；被省略算法不藏在清理diff中。外围导入、配置、数据类、AST加载器不是被省略算法的可运行替代品。

PiToMe非商业许可、DyMU MIT及LLaVA组件Apache、IPCV根Apache、InternVL底层MIT、HF CLIP/Qwen版权归属分组件保留；未给全仓统一MIT。此次HF CLIP来源说明已经补入新快照，首批文件中“暂缓clip_hf.py”的说明保留为首批审查记录。两批目录并存，不用新副本冒充未改字节。

Qwen strict Dense failed_preserved、根因未定、Qwen DToMe标量阈值变体及Smol DToMe dispatch/none-pass差异仍原状披露。Smol iLLaVA保留质量来源映射150点及VERIFIED12的边界见首批16组说明；本批没有新质量、时延、GPU或模型实验。

没有真实基准题干、原始预测/日志或个人绝对路径。prompting.py含公开LLaVA通用对话模板，属于源代码模板，不是本用户会话或基准实例。

## 清单与既有文件

基线commit为 ed7b06b65c5e563106671e76fe37ce4e0f2cac78；原有root README和首批29文件逐blob核验后保留不变。本批原ZIP的根MANIFEST不覆盖仓库原MANIFEST，重复的3份许可证也不重复改写。新的完整仓库视图见 [统一公开清单](sources/PUBLIC_REPOSITORY_MANIFEST_20261009_r2.json)，包含基线全部文件和本批增量，清单本身除外。根MANIFEST仍是首批快照的范围清单。
