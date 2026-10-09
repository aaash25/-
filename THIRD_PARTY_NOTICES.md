> 范围说明：以下保留首批公开审查记录，其中源码数量和暂缓文件描述只针对首批。当前已追加迁移源码快照及许可补充；请以[仓库导航](README.md)和当前组件说明为准，不把首批范围误读为当前仓库没有源码。

# 第三方来源与许可边界

PiToMe 来自 [hchautran/PiToMe 固定版本](https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40)。本次仅收入 merge.py、utils.py 两份参考文件。其 Meta Platforms, Inc. and affiliates 原版权头原样保留。PiToMe 作者及引用见 [参考说明](third_party/pitome/README.md)。根 LICENSE 的 CC BY-NC 4.0 与非商业条件继续适用，详见 [原许可文本](third_party/pitome/LICENSE) 与 [官方条款](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)。setup.py 的 MIT 元数据不能覆盖根许可；本仓库不将这些文件改授 MIT，也不暗示上游作者背书。

三份原许可证与两份参考源码保留候选中的原字节。本次按固定 GitHub blob 复核，所有六份原候选第三方文件（包括未选入的 clip_hf.py）与上游仅有 CRLF/LF 换行差异，无其它字节变化；入选文件的候选 SHA256、上游 blob SHA、上游原始 SHA256 和变换说明见 [逐文件出处](sources/SELECTED_FILE_PROVENANCE.json)。这不是“与上游原始字节完全相同”的声明。

DyMU 根 MIT 和 IPCV 根 Apache-2.0 许可正文、版权文字分别保留，仅作为来源许可记录；本次未收入二者方法源码。以后选入依赖或嵌入组件时应各自保留适用声明，不能以根许可覆盖全部组件。

iLLaVA 固定版本未发现方法根级 LICENSE，本次只提供独立说明和上游链接。PiToMe clip_hf.py 待补齐内嵌 Transformers 的 Apache 许可和上游原版权/归属说明，暂未收录；保留双方适用许可并补齐声明后可重新审查，不是已确认禁止公开。混合迁移文件尚未逐文件闭合来源/许可链者继续排除。

没有给整个仓库统一赋予 MIT；这些检查不是全部第三方权利的法律认证。
