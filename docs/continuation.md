# 工作接续状态

进行中（2026-11-01）：新轮 completed-protection-v1 已在 feat/completed-candidate-protection 设计/实现，24 开发源句和保护规则已保存。先真实开发验证并冻结，再跑同16/40的四融合控制（原始、旧约束回退、旧早停回退、新候选保护回退）Base→So400m。新源/实测尚未完成，不能把下文上一轮结果当本轮结果。台账 SIG-051/052 记录当前边界。

更新：2026-11-01。本轮重复检测与提前拒绝实验已完成，没有训练。仓库为 `/Users/caochao/Documents/SigLIP`，私有远端为 `HenryVarro666/siglip-embedding-tool`。代码、两模型实测、独立复算、真实 CLI、打包、干净克隆与 515 项测试均已完成。代码检查点为 `053caa86417072f434b894173a2a86bf68b3b0b2`；后续从 `main` 接续，最新文档提交的检查以 GitHub 对应提交的 Actions 为准。

## 从这里开始

[文档索引](README.md) → [本轮完整报告](repetition-early-stop-v1.md) → [开发与复现/API](../benchmarks/repetition-early-stop-v1/README.md) → [问题 SIG-044–050](issue-log.md) → [验收凭证](../benchmarks/repetition-early-stop-v1/verification.json)。

用户要求过去、现在、未来每次问题与改进持续保存。发现即登记稳定编号、证据、影响和下一步；修复后补实际验证与剩余边界，结束更新本页、索引和改进路线。规则已写入 AGENTS.md。旧报告与负结果不得覆盖。

## 结论与当前默认

同一批已诊断的 16 条查询、40 张图，比较 3 个中英融合方案：原始 OPUS、现有约束失败后回退原始 OPUS、新增重复早停后回退原始 OPUS。每策略每模型 3 次预热、16 次完整评分及 5×16 次热请求，共 96 次评分、480 次热请求。每次真实翻译，未缓存译文；完整耗时包含先失败尝试、检测、异常清理、备用翻译及检索。

| 策略 | Base mAP / 完整 p50 ms | So400m mAP / 完整 p50 ms |
|---|---:|---:|
| 原始 OPUS 融合 | 0.7978 / 89.84 | 0.8534 / 113.65 |
| 现有约束→原始 OPUS 融合 | 0.7614 / 433.25 | 0.8221 / 506.81 |
| 重复早停→原始 OPUS 融合 | 0.7614 / 449.96 | 0.8221 / 472.90 |

新方案主拒绝率由 6/16（37.5%）增至 7/16（43.75%），最终失败仍为 0/16，实际备用成功 7/7。早停 t09/t12/t13，触发步数为 60/60/36；只捕捉旧 6 条失败中的 t12/t13，另 4 条漏检。t09 为旧机械检查接受、新方案早拒绝的反例，不能把“活动 beam 重复”当作全部候选无效或语义错误的证明。

同旧 6 条失败的完整平均耗时降低约 4.27% / 10.37%，但全 16 条的平均耗时 Base +0.61%、So400m −7.02%；整体收益不稳定。新旧聚合检索指标相同，t09 实际英文已有改变，不声称所有分数完全相同。**原始 OPUS 融合仍为默认，早停作为显式实验选项。**

## 开发冻结与不可改写的证据

1. 阶段 1 先声明严格后缀 16 候选，8 条开发查询中 7 成功、1 条 i07 无 EOS；全部候选未捕捉失败。完整负结果保存在 `7fb0973` 与 `benchmarks/repetition-early-stop-v1/development-stage1/`。
2. 仅依据开发观测修订有界尾部检测，先在 `2802893` 声明新网格，再重新观测同 8 条开发查询。最终 profile `bounded-n16-r2-min32-q4-tail4` 于 `905f130767d019a4bcd705f80d3d50ad0eba0924` 冻结：32 token 起每 4 步、4–16 token 周期重复至少 2 次、最多 4 个尾 token、4/4 活动 beam 共识；64 上限不算提前。
3. 实际开发 i07 在 48 步抛拒绝异常并新运行原始 OPUS。7 条成功例都没进入选中规则的 32 步检查区间，因此不能由零误拒绝推出长正常查询安全。开发覆盖缺口见 SIG-048。
4. 正式协议于 `b6d0aec1f64310913517a6f27e6c70d95a0eb7bf` 冻结后才运行本轮 16 条。没有据正式结果重调规则。历史正式结果原已知；当前数据不是新的独立现场留出，标签仍非人类领域真值。

精确摘要：

- 新测量核心 SHA256：`5508fc7372887770f584065949d19debe77b81fd6d09cf022a432de164d7bc12`。
- 正式协议 SHA256：`cfdc70f0d5c75bb5cd2fb242c5eaed033b7b7c975f72f273028659c13ab59229`。
- profile 字节 SHA256：`e93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777`。
- 旧回退核心 `e377efde…`、旧术语核心 `44b19b06…`、旧 CLI、词表、权重、解码器均保持不变。原始损失示例与备用训练代码保留，没有启动预训练模型训练。

## 路径与验证

- 代码：`src/siglip_tools/repetition_stop.py`、`early_stop_development.py`、`early_stop_benchmark.py`；新增模块入口显式启用，旧默认接口不变。
- Git 内证据：`benchmarks/repetition-early-stop-v1/` 的冻结输入、两阶段开发记录、profile、results-summary.json、requests.json、audit.json、verification.json。requests 保存完整评分排名/原样轨迹及每次热请求计时与可重建摘要。
- 本机原始结果：`artifacts/repetition-early-stop/{base-v1,quality-v1}/result.json`。Base SHA256 `d728d9412ab1410c71aca46cd5f448a1fdc33dc057abc7bda33606dd6763f894`；So400m `1ec659c53182e8d5112dbee0c092b7ec42af81910b025806ed5a415a00929ba2`。
- 原图 `data/equipment-v1/images/`；索引 `artifacts/equipment-benchmark/{base-v1,quality-v1}/index.npz`；OPUS/解码器 `models/opus-mt-zh-en`、`models/opus-constrained-decoder`。这些均不上传 Git，两模型索引不可混用。
- 独立记录复算通过 404,544 项检查、106,488 次数值比较，最大差 `2.842170943040401e-14`。包括开发选择、触发证据、指标/失败率、全部计时、控制与跨模型一致性；不是独立神经重放或人类语义审核。
- 本地、独立克隆及[代码检查点 CI 41281069556](https://github.com/HenryVarro666/siglip-embedding-tool/actions/runs/41281069556) 均 515 通过、2 项真实模型测试默认排除，无 skip；Ruff 与原始 CLIP/SigLIP 有限差分演示通过。克隆复用本地虚拟环境及已固定的外部解码代码缓存，71 个源/测试/示例/训练/构建/CI 文件字节一致。
- 真实 MPS 模块 CLI 的 t13 真早停、t09 旧接受早停、t14 无约束和 t01 正常约束通过，轨迹/Top5 与正式 Base 一致；仅功能验证，不混入性能统计。wheel/sdist、包内模块/profile 字节及仓库外导入通过。
- SIG-050：首次审计与 pytest 日志写入并行，科学数值均通过但一个日志摘要尚未闭合。等所有写入结束后顺序重审计，数值/源结果完全不变，全部日志摘要一致；初版快照保留。以后日志 glob 必须依赖所有写入者完成。

最新主分支提交与逐提交 CI 见[提交历史](https://github.com/HenryVarro666/siglip-embedding-tool/commits/main/)和[Actions](https://github.com/HenryVarro666/siglip-embedding-tool/actions)。固定凭证绑定上述代码检查点，不能把旧 CI 冒充未来 HEAD。最终上传凭证保存在本机 `artifacts/repetition-early-stop/publication.json`。

## 下次执行入口

```bash
cd /Users/caochao/Documents/SigLIP
git status --short --branch
uv run pytest -m 'not integration'
uv run python -m siglip_tools.repetition_stop search \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --query '变压器的散热片在垂下的黑色粗电缆后方，电缆连接顶部的套管。' \
  --mode auto_fusion --fallback opus --device mps --local-files-only
```

实测已完成，不必为了继续阅读重跑模型。比较、导出与复算命令见本轮 README，输出存在时拒绝覆盖，使用新路径。先等全部日志写完，再收集文件摘要。

下一轮候选工作尚未执行：补长正常句、合法重复和列表开发控制；研究如何感知/保护已完成且满足约束的候选；分析 4 条漏检的循环/停滞形式；领域人员复核术语与接线/否定/安装关系；另立新版本和独立现场留出。耗时需交错模型/时段、多会话及 CPU/GPU/内存 profiling，当前没有缓存、OCR、局部特征、并发服务或训练。不要据本 16 条修改当前阈值或回写标签。

旧进展均保留：[设备](equipment-benchmark-v1.md)、[中英融合](bilingual-fusion-v1.md)、[自动翻译](auto-translation-v1.md)、[更大译者](stronger-translation-v1.md)、[术语约束](terminology-constraints-v1.md)、[失败回退](constraint-fallback-v1.md)。上一轮完整接续在 Git `40e29ac`；历史 SIG-001–019 与后续 SIG-020–050 持续衔接。
