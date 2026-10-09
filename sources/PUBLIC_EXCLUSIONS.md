# 公开排除范围

不收入私人审核 ZIP、真实基准 prompts/messages、原始质量或 timing journals、预测答案、个人绝对路径、服务器连接资料、凭据、传输助手、模型权重、原始数据集和图片。来源不明的多方法模块与 iLLaVA 代码不收入公开候选。当前仅公开状态说明、出处和许可证，以及两份 PiToMe 上游参考文件，不提供完整迁移运行环境。

PiToMe 的 algo/pitome/patch/clip_hf.py 暂未选入：固定 PiToMe 版本的 forward 与 Transformers v4.37.0 CLIPEncoder.forward 存在大段相同源码，而该参考文件没有保留相应 OpenAI/HuggingFace Apache 版权头。仅有 PiToMe 根 CC 许可不能替代这条内嵌来源的声明/许可核对。待补齐适用的 Apache 许可、上游原版权/归属说明及 PiToMe 许可边界后，可重新审查收录。此暂缓不表示已确认代码禁止公开，也不是算法错误判定；不以添加泛化 NOTICE 代替逐文件核清。

完整运行配置和结果 JSON 均未公开。本次新增文件不包含根 README.md，不覆盖既有根说明；原始私人证据与方法源码不改写。源文件复制仅限白名单。
