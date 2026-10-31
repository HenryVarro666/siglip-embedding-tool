# 工作接续状态

更新：2026-10-31；当前任务进行中，完成时将更新下文。目录：`/Users/caochao/Documents/SigLIP`，私有远端 `HenryVarro666/siglip-embedding-tool`，开发分支 `feat/opus-terminology-constraints`。

## 当前授权与固定边界

给现有 OPUS-MT 增加逆变器、变压器套管、可见散热鳍片约束；用16个新增中文意图检查接线、安装和否定关系，对比原始beam1、无约束beam4、约束beam4在Base/So400m的效果及端到端时间。继续不训练，不手改英文，不用新查询调词表/参数。新增查询共享原40图库，助手标注，不能称为现场独立人类真值。

用户额外要求：历史、当前、未来每次问题/改进均保存到文档并衔接。先登记[问题台账](issue-log.md)，修复补验证，结束更新此文件与[总索引](README.md)。历史19组问题已回填，不覆盖旧报告。

## 已完成

- 历史主分支 `15c9822`：NLLB对照完成，无训练，259项离线测试/构建/CI通过；[报告](stronger-translation-v1.md)保留。
- 新设计与初始台账提交 `38bf2fd`；词表与上下文规则在新查询编写前冻结提交 `3e0da59`，glossary SHA256 `20fef944133fe8453e9b60d5d7a6ccabb93ce3a4ffe0565038caaf8f51e3c42e`。
- 新查询由未看历史查询/输出的助手检查40原图后编写，queries SHA `19a1da3d4fa9d961f379af0d56dad9274e698895af0d2800d35ad44dfad445db`，qrels SHA `0a6c9486084c5108439f642f4440d8ec4191ff8221a2ee0c553f8589f12daf95`。根执行器确认与旧文本无完全相同项；第二助手正在复核标签。
- OPUS适配器37项测试通过，含固定上游生成实现+随机微型Marian的真实接口；无外部源缓存时22项明确跳过、15项仍离线通过。CI增加显式准备四个代码/README文件，不下载预训练权重。旧代码默认保留。
- 新七策略runner与CLI的14项测试通过，含失败仍计入分母、完整阶段/真实总时间、比较入口子进程和约束轨迹。尚待真实旧查询预检、协议/代码冻结、两检索模型运行、独立复算及最终全套/CI。

## 固定来源与本机路径

- OPUS `Helsinki-NLP/opus-mt-zh-en` rev `cf109095479db38d6df799875e34039d4938aaa6`；模型 `models/opus-mt-zh-en`。
- Base索引 `artifacts/equipment-benchmark/base-v1/index.npz`；So400m索引 `artifacts/equipment-benchmark/quality-v1/index.npz`。索引记录各自完整模型/processor身份，禁止混用。
- 外部解码源 `transformers-community/constrained-beam-search` rev `07b2f120b7db38f1d7bac617ad65ea130508f297`；准备命令 `uv run python -m siglip_tools.opus_constraints prepare --output models/opus-constrained-decoder`。其代码不入Git，完整固定SHA和本地来源清单由适配器验证；准备阶段只取四文件。
- 新评估输入 `benchmarks/terminology-constraints-v1/`，图库仍为 `data/equipment-v1/`；新原始结果/日志放 `artifacts/terminology-constraints/`。旧实验目录不可当作新结果输出。

## 下一步

1. 第二助手完成标签复核，补标签审阅SHA；准备固定解码器；只用旧8条开发查询做真实接口预检，所有问题/修复入台账。
2. 固定完整协议（7策略、16意图、每策略预热3次/每意图10次、float32/MPS），冻结代码/索引/词表/来源/查询/标签哈希并提交。
3. 顺序运行 Base / So400m；保存未修改英文、约束轨迹、完整排序、全部热计时与失败，不从指标分母删失败。
4. 文本/上下文审阅独立于检索分数；复算结果，保存报告、可复现摘要、Top5页和改进点。仅8条旧开发查询上的调试不算新集效果证据。
5. 全套离线测试/lint/原始损失演示/构建（含词表资源）/独立检出，代码及结果审阅后提交、私有推送、验证该head的CI；最后更新此接续文件。

如果会话中断，先核实 `git status`、本文件和计划进度，不重复已完成的模型运行或覆盖既有结果。未完成的真实运行不能写成完成，静态词覆盖不能代替关系保真结论。
