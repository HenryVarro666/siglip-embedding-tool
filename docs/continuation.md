# 工作接续状态

更新：2026-11-01。最新“已完成候选保护”实验已完成，无训练。仓库 `/Users/caochao/Documents/SigLIP`，私有远端 `HenryVarro666/siglip-embedding-tool`；后续从 `main` 接续。源码/冻结检查点 fa8eddf 已上传且 CI683 通过，最新文档/结果提交须查看对应逐提交 CI，不能用旧回执冒充未来 HEAD。

## 从这里开始

[文档索引](README.md) → [最新完整报告](completed-protection-v1.md) → [开发/复现/直接调用](../benchmarks/completed-protection-v1/README.md) → [台账 SIG-051–061](issue-log.md) → [验收凭证](../benchmarks/completed-protection-v1/verification.json)。用户要求之前、现在、未来每次问题和改进持久化；发现即登记编号/证据/影响/下一步，修复写实际验证及剩余限制，每次结束同步本页、索引和改进路线。AGENTS.md 已固化。

## 最新结论

先声明24开发句（原8、8长正常、8合法重复），实际17条旧约束接受、其中10条新增长/重复控制进入≥32检查。r02在34步已有真实EOS候选，旧早停56步拒绝，新保护在56/60步继续至64并保留旧输出；开发17旧接受无新误拒绝。再用同一已诊断16查询/40图比较四个融合控制，每模型每策略3预热、16完整评分及5×16热请求，共128评分/640热请求；全部真实翻译/备用，无译文缓存。

| 方案 | Base mAP / 完整p50 ms | So400m mAP / 完整p50 ms |
|---|---:|---:|
| 原始 OPUS 融合 | 0.7978 / 89.18 | 0.8534 / 110.92 |
| 旧约束失败→OPUS | 0.7614 / 413.69 | 0.8221 / 444.83 |
| 旧重复早停→OPUS | 0.7614 / 429.80 | 0.8221 / 442.53 |
| 完成候选保护早停→OPUS | 0.7614 / 411.18 | 0.8221 / 445.23 |

新保护t09在60步抑制拒绝，最早真实EOS32步、继续64并返回32-token候选；主拒绝7/16→6/16（43.75%→37.5%）、实际早拒绝3/16→2/16，新增早拒旧接受0/10，最终0/16；备用6/6成功。仍提前捕捉t12/t13（60/36步），仍漏t02/t04/t10/t11。两模型集合相同。

整体均值相对旧早停 Base+0.69%、So−0.91%，p50−4.33%/+0.61%；t09组均值降低约6.75%/4.15%，总体不构成稳定加速。推荐基线继续原始OPUS融合，保护器显式启用，原有默认接口未变。词汇/EOS保护没有修复筋条/导线、否定、接线/安装或对象归属语义；同16为已知工程回归和助手标签，不是新独立现场验证。

SIG-059更正：上一轮报告误把t09失败诊断片段变化写成最终检索英文变化。核对旧两模型原数据及本轮，新旧最终英文/query_texts/排名分数相同，改变的是诊断、主路径和备用调用。旧报告已附日期更正，原数据/协议/计时不变；没有逐维保存/核对embedding的额外声明。

## 精确版本与协议

- 源句/固定结构规则预声明 `70cf127`；测量源代码 `768ea626c555d6214ceca04fab24b8ee992b02b2`。
- 开发协议 `5e527747c7944195466e1347b243d474303c63ab`，开发实测冻结 `150558cdfc12bd52be694fadbf45e01e5198a340`。
- 正式冻结 `fa8eddfb3245ab5bc6f40f07adc3a753e22d6cae`，新核心 SHA256 `b120dd7193236fdbb3f965274b45436d2d2c3008f9d3b1bf3fbcbd55b61ee2ad`。
- 正式协议 SHA256 `b60158caa27b6855ed995cdc956cb5b6f89bf59f369ed560ccf883c9e43324fc`。
- 新规则字节 SHA256 `a8fb1d276fc53ace8c59b76dabd13a24035c323aa637f184c768b20db2464cdc`，原重复profile `e93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777` 不变。旧5508/e377/44b核心、CLI、词表、权重、外部解码源及原学习/训练备用代码保持。

固定保护仅在当前仍保留、process中实际选中EOS且全部token/完整词形/非空通过的候选存在时抑制重复拒绝，继续原搜索/finalize；不补救/替换候选，不把finalize补EOS当真实完成。请求局部函数/类，不改全局/源码；记录EOS提案/得分/成员、在线词检查和实际abort。最终接受与精确回调覆盖已按SIG-054补验证，scorer_done_before_finalize只在finalize前读取一次。

## 代码、结果与验收

代码 `src/siglip_tools/{completed_protection,protected_development,protected_benchmark}.py`；完整输入/开发/协议/摘要/requests/复算/验收在 `benchmarks/completed-protection-v1/`。requests保存128次完整评分及640次逐热计时/可重建摘要。当前旧3控制与上轮两模型的英文、轨迹、路由及排名分数完全一致，当前时延全部重新实测。

本机原始结果 `artifacts/completed-protection/{base-v1,quality-v1}/result.json`：

- Base SHA256 `fc63e57520cbe6ea98a549812e3d020d758167e785a32d26c93cdba92931f2a4`。
- So400m SHA256 `fa531354d035c1fefe720444e0b51590aa5da6584bb6604a144bbfae21a2d30e`。

原图仍 `data/equipment-v1/images/`；索引 `artifacts/equipment-benchmark/{base-v1,quality-v1}/index.npz`；模型/代码缓存 `models/opus-mt-zh-en`、`models/opus-constrained-decoder` 及固定HF缓存。均不上传Git，严格匹配索引/model/processor/runtime身份，不以相同维度绕过兼容检查。

实际验证完成：

- 独立开发复算138335检查/51034数值，正式880301检查/257832数值，最大差0；另有只读终审890项记录核对，无可操作发现。复算不等于独立神经重放、重分词或人类语义真值。
- 本地/干净克隆683通过、2真实模型测试默认排除、无skip；Ruff和原始CLIP/SigLIP有限差分演示通过。82个源/测试/示例/训练/构建/CI文件与检查点相同。克隆复用已装虚拟环境及固定外部解码代码缓存，备份训练测试只用玩具fixture，本轮没有预训练模型训练。
- 真实MPS模块CLI：t09保护、t13真早拒绝→OPUS、t14无约束、t01正常约束，完整轨迹/Top5匹配正式Base；功能验证不并入性能样本。
- wheel/sdist、模块/两profile/词表字节及仓库外导入/帮助通过。python -m build入口不可用的初次尝试保留，使用已装uv离线构建，未安装/升级推理依赖。
- 源码/冻结检查点[CI41283470859](https://github.com/HenryVarro666/siglip-embedding-tool/actions/runs/41283470859)实际683通过。最终文档/结果HEAD另有CI，见[提交历史](https://github.com/HenryVarro666/siglip-embedding-tool/commits/main/)和[Actions](https://github.com/HenryVarro666/siglip-embedding-tool/actions)；最终上传凭证为本机 `artifacts/completed-protection/publication.json`。

所有检测、观察/CPU复制、在线词检查、先失败/清理、实际备用、检索和返回轨迹复制都在报告的外层同步总时间内；完整记录科学重放和排名/JSON整理放在计时后（SIG-056）。全部尾样本保留，无模型失败尝试被删除/排除，无为改善分数重跑。等所有日志写完后才作审计摘要，SIG-050流程没有复发。

## 下次入口与尚未执行的方向

```bash
cd /Users/caochao/Documents/SigLIP
git status --short --branch
uv run pytest -m 'not integration'
uv run python -m siglip_tools.completed_protection search \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --query '变压器的散热片在垂下的黑色粗电缆后方，电缆连接顶部的套管。' \
  --mode auto_fusion --fallback opus --device mps --local-files-only
```

阅读报告不必重跑模型。比较/导出/复算和自定义开发路径见本轮README，现有文件拒绝覆盖，重跑用新目录；共享日志摘要必须等全部写入结束。

下一轮仍未执行：真实现场/人类审核留出；EOS出现前的合法重复与词汇合法但坏语义候选；4条漏检；软约束/源槽位/其他预训练翻译；多会话交错模型/时段、CPU/GPU profiling、内存峰值/并发/大库。任何方法/数据改动另建版本，不据本16调当前阈值，不隐式加训练、OCR、局部特征或缓存。用户若仅要求继续当前任务，先核对Git和本页，不重跑已完成实验。

旧进展保留：[设备](equipment-benchmark-v1.md)、[双语](bilingual-fusion-v1.md)、[自动翻译](auto-translation-v1.md)、[更大译者](stronger-translation-v1.md)、[术语约束](terminology-constraints-v1.md)、[失败回退](constraint-fallback-v1.md)、[原重复早停](repetition-early-stop-v1.md)。上一轮完整接续在Git7386423；历史SIG-001–050与新SIG-051–061持续衔接。
