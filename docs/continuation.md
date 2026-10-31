# 工作接续状态

更新：2026-10-31。本轮测量、分析和本地核验完成，正在做独立检出与私有发布核验。目录：`/Users/caochao/Documents/SigLIP`，私有远端`HenryVarro666/siglip-embedding-tool`，开发分支`feat/opus-terminology-constraints`。

## 先读什么

[文档总索引](README.md) → [本轮完整报告](terminology-constraints-v1.md) → [问题台账SIG-020–037](issue-log.md) / [历史SIG-001–019](historical-issues.md)。用户要求之前、现在和未来每次问题/改进都保存并衔接；发现即登记，修复补实际验证，结束更新此接续文件和索引。要求已写入AGENTS.md，不能只留聊天。

## 本轮固定内容与结果

- 继续不训练。OPUS原始beam1、普通beam4、领域约束beam4，与Base224/So400m384对照；每次自动请求真实翻译，英文不手改，融合各0.5余弦。原始默认保留，约束显式可选。
- 词表/上下文规则先冻结提交`3e0da59`；16个新中文意图、32配对字符串在无新输出时编写/复核，仍共享旧40图库，助手标签而非人类/现场真值。源英文字段不用于翻译。
- 测量代码提交`b68cb99`；核心SHA`44b19b06d0ffd628ea48ef0bb7ec69c64025637e0e664e43001efdc393ded9df`，协议SHA`807dddddef6282aa57e93afc7754a40e67bb54f919c8b33296cd63f3e9c806b2`。完整输入/来源/索引/代码绑定见`benchmarks/terminology-constraints-v1/freeze.json`，旧报告不覆盖。
- 两模型7策略均完成：2240热请求含240失败；224评分尝试为200完整40图排名和24失败。两模型约束均失败t02/t04/t10/t11/t12/t13，共6/16，全部缺EOS；不从分母删除。
- 原始/约束融合mAP：Base0.7978/0.4292、So400m0.8534/0.5469；实际端到端p50为100.79/582.64ms和119.52/535.95ms。词义和局部成功查询有改善，未抵消失败/关系损失，不支持设为默认。
- 匿名文本审阅后解盲：约束术语明确9/16（含失败片段），原始6/16；套管4处均没有完整可用译文。接线/安装等关系仍有错误。t16源文歧义仅作事后敏感性，原查询/标签不改。
- 独立复算通过；完整证据、原样输出/失败序列、匿名与解盲文件、结果摘要、审计/复算程序均已保存。SIG-037修复事后导出器绑定，真实重导出除时间戳外完全一致，未改测量核心或重跑模型。

## 验证与证据位置

- 冻结前315离线测试/Ruff通过；最终本地319通过、2真实模型测试默认排除，Ruff、原始CLIP/SigLIP损失/有限差分演示、wheel/sdist/包资源检查通过。日志`artifacts/terminology-constraints/final-verification-v2.log`。
- 真实MPS CLI约束t01成功，英文/轨迹/Top5在`artifacts/terminology-constraints/cli-smoke.json`。源参数、规则、权重没有训练或手改。
- 原始结果：`artifacts/terminology-constraints/{base-v1,quality-v1}/result.json`；对照`comparison.json`。审查页`artifacts/terminology-constraints/review.html`：1360卡片、24失败区，中文控制重复仅为展示。
- Git内输入、摘要、译文与审阅：`benchmarks/terminology-constraints-v1/`；外部解码来源四文件在`models/opus-constrained-decoder`，OPUS在`models/opus-mt-zh-en`；Base/So索引仍在`artifacts/equipment-benchmark/{base-v1,quality-v1}/index.npz`。
- 独立检出与远端状态尚待收尾，不能据本地绿测试假定已推送。完成后本文件会删除此待办；当前真实远端提交/CI以[提交记录](https://github.com/HenryVarro666/siglip-embedding-tool/commits/main/)和[GitHub Actions](https://github.com/HenryVarro666/siglip-embedding-tool/actions)为准。

## 后续建议（未执行，不训练）

1. 人工审核t16柜外/柜内位置、e026同图两设备归属、接线/连接/鳍片的领域含义；纠正另建数据版本并重评，不覆盖v1。
2. 在其他开发查询预定重复抑制、预算/beam、源槽位/短语或软约束，再用新留出集测。显式失败回退要记录触发率、先失败再回退总耗时；本轮未实现回退/缓存/新生成策略。
3. 现场查询/实物分组与模型顺序交错、并发/峰值资源/profiling待补；词出现、句意忠实、请求可返回和检索命中分开评价。
4. 如需要，预声明检索权重字节SHA，存逐重复输出/排名或摘要；本轮审计边界见SIG-035。上游processor提示保留，不以身份一致冒充已定位影响。

任何后续修改先记录问题编号、证据与作用范围，再改代码/数据；全套测试和真实测量各有自己的口径，未执行的工作不能写作完成。不要重复已存在的模型运行或覆盖结果。
