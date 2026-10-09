# PiToMe 上游参考摘录

选定来源：[hchautran/PiToMe](https://github.com/hchautran/PiToMe/tree/550b5deed94aadfeac28bfbe381d87e672044a40)，commit 550b5deed94aadfeac28bfbe381d87e672044a40。

PiToMe 工作 Accelerating Transformers with Spectrum-Preserving Token Merging 的作者：Hoai-Chau Tran、Duy M. H. Nguyen、Duy M. Nguyen、Trung-Tin Nguyen、Ngan Le、Pengtao Xie、Daniel Sonntag、James Y. Zou、Binh T. Nguyen、Mathias Niepert。作者项目说明见[固定 README](https://github.com/hchautran/PiToMe/blob/550b5deed94aadfeac28bfbe381d87e672044a40/README.md)，论文见 https://arxiv.org/abs/2405.16148 。上游明确致谢 ToMe；入选两文件中 Meta Platforms, Inc. and affiliates 的版权头予以保留。

本目录只包含 algo/pitome/merge.py、algo/pitome/utils.py 和原 LICENSE。它们用于非商业研究参考，不是任一模型已经迁移完成或可直接安装运行的发行包。文件不加载模型、不被本说明执行。

适用上游根 CC BY-NC 4.0；完整条款及免责内容见 [LICENSE](LICENSE)。未修改算法或文本内容；归档副本为 CRLF，上游 Git blob 为 LF，两者规范化换行后逐字节一致，原始双哈希见 ../../sources/SELECTED_FILE_PROVENANCE.json 。不采纳 setup.py 中冲突的 MIT 标签。

algo/pitome/patch/clip_hf.py 暂未收录，原因见 ../../sources/PUBLIC_EXCLUSIONS.md 。本摘录不声称所有上游依赖已获本仓库重新授权。
