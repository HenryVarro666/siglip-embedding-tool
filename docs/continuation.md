# 工作接续状态

追加进行中（2026-11-01）：重复检测/提前拒绝新轮，开发分支feat/repetition-early-stop。阶段1严格后缀16候选全未抓住开发i07，负结果保存在7fb0973及benchmarks/repetition-early-stop-v1/development-stage1；根据开发beam尾部术语插入现象，阶段2在2802893先声明有界尾部规则，再用同8开发查询观测/选择。最终profile在905f130冻结：bounded-n16-r2-min32-q4-tail4，四beam共识，min32、每4步、周期4–16重复2次、容忍4尾token；真实开发i07在48步中断并原始回退，7旧机械成功未早拒绝，但它们没有进入最终32步检测区间，不能当长句误拒绝率证据。

正式3融合策略（原始、现有约束→原始、新早停约束→原始）协议已于b6d0aec在新16输出前冻结，核心SHA5508fc7372887770f584065949d19debe77b81fd6d09cf022a432de164d7bc12、协议SHAcfdc70f0d5c75bb5cd2fb242c5eaed033b7b7c975f72f273028659c13ab59229、profile字节SHAe93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777。496离线测试通过/2排除，Ruff通过。Base已完整完成，原始/旧回退/新早停mAP0.79783/0.76137/0.76137、p50 89.84/433.25/449.96ms；早拒t09/t12/t13、捕捉旧失败2/6，t09为旧机械接受反例，规则未调。So400m正在artifacts/repetition-early-stop/quality-v1.log运行；不要覆盖已有result或同时跑另一GPU模型。新结果/独立复算/完整报告与最新验收仍待完成。台账新增SIG-044–049。下文是上一轮已完成证据，结束本轮后改成本轮完整接续。

更新：2026-11-01。最新约束失败回退实验已完成两模型实测、独立算法复算、本地401项离线测试、真实MPS CLI及打包检查；最终检出/上传凭证以本页末尾和verification.json为准。仓库`/Users/caochao/Documents/SigLIP`，私有远端`HenryVarro666/siglip-embedding-tool`，开发分支`feat/constraint-failure-fallback`，最终从主分支main接续。

## 从这里开始

[文档索引](README.md) → [本轮完整报告](constraint-fallback-v1.md) → [输入/复现](../benchmarks/constraint-fallback-v1/README.md) → [台账SIG-038–043](issue-log.md)。用户要求过去、现在和未来每次问题/改进都持久化：发现即登记，修复写实际验证与剩余边界，结束更新本接续和索引；AGENTS.md已固化规则。

## 本轮完成内容

- 不训练。同一已诊断16查询/40图，9策略、每策略3预热/16完整评分/5×16热请求。288评分、1440热请求；不回退控制24评分/120热请求最终失败，四回退策略48评分/240热请求触发均返回；主拒绝仍六条6/16、30/80，不改成0。
- 真实先约束，再按配置重新原始beam1 OPUS或中文一次；无英文缓存、不按ID跳过、不检索失败片段、不隐式级联。完整总计时包括先失败成本。CLI新增--constraint-fallback error|opus|chinese，原始默认保留；显式预期最终失败输出JSON并退出2，系统/默认error仍stderr错误。
- 原始/约束→OPUS融合：Base mAP0.79783/0.76137、p50 99.56/551.35ms；So400m0.85337/0.82212、117.66/656.95ms。约束→中文融合入口0.71176/0.76179、642.12/551.37ms。最终可返回恢复没有修复已接受的语义/排序，当前原始融合仍更好更快。
- 748,678项独立算法检查、131,907数值比较通过，最大统计差1.14e-13。跨模型MT原样英文/完整轨迹、各分支控制/历史控制、全部评分与热摘要/时间均核对，重复/分支分数漂移0。程序由另一助手未看本轮输出时编写，真实运行由主执行器完成；不是独立神经重放或人类真值。
- 用户中断后确认Base已完整保存，未重跑；So恢复后单独完整运行。模型间约4.5小时暂停、未控制热状态/时段和尾延迟保留，见SIG-040/042。主分母16不排除t16，不修订旧标签/词表/beam/长度。

## 精确版本与路径

- 测量代码d005f31ccacc52e7567514f70ae594d4c542e296；协议/冻结提交71ead7f9ebbc71ba8aa5e2d93f413642d29ab98f。后续导出/独立复算与测试代码检查点2f2e84488905b2dbed9caef8e24c8ce642daa3b3。
- 核心SHA e377efdeb1bcfe1ff44ac93eca1ee8489d1a5a2032fcba0eabbfdeae958f8675；协议SHA b4e64cd4e9a7314a227386c62507363f22c594a9d8856299301bbca4e175efc1。旧测量核心44b19b06…与旧报告保持不变。
- 输入、protocol/freeze、来源、results-summary.json、requests.json（完整评分排名/轨迹+全部热计时/摘要）、analysis.json（6/10组及尾样本）、audit.json、verification.json：`benchmarks/constraint-fallback-v1/`。
- 原始结果`artifacts/constraint-fallback/{base-v1,quality-v1}/result.json`；Base SHA a7325d62983472e60d820d19cf4fb96daf3318a03b90397dddef6ebfeb86e566，So SHA 7cd928a22159baf8acf7e6917675451480a1f66569547f6c99f4b2a32e2f95d2。日志同目录；恢复记录resume-event.json；CLI、测试、CI、build/wheel凭证也在此忽略目录。
- 图库仍data/equipment-v1；索引仍artifacts/equipment-benchmark/{base-v1,quality-v1}/index.npz。两套索引不可混用，identity/revision/processor/runtime/文件SHA必须匹配。OPUS/解码器在models/opus-mt-zh-en、models/opus-constrained-decoder，不上传权重或外部源。

## 验证状态

本地401通过、2真实模型测试默认排除、无skip；Ruff和原始损失/有限差分演示通过。真实MPS CLI的t02→原始、t02→中文及t01成功不回退，轨迹/Top5与正式结果一致；这些功能验证不加入性能统计。wheel/sdist构建、模块/词表字节核对和仓库外导入/帮助通过。

当前审计修复/代码验收提交c33476d4f7b230a6b3a17c009f15ba0a15206227已上传私有仓库，[CI41276088083](https://github.com/HenryVarro666/siglip-embedding-tool/actions/runs/41276088083)成功；本地及该提交独立检出均401通过/2排除，所有56个源/测试/示例/训练/CI文件与该提交相同。SIG-043仅修复审计活动日志摘要，新版复算原数值完全不变；旧400项验证保留为历史，不冒作当前401覆盖。证据见verification.json。最终文档/凭证提交不冒用旧CI为未来HEAD结果，最新真实main及per-commitCI看[提交](https://github.com/HenryVarro666/siglip-embedding-tool/commits/main/)与[Actions](https://github.com/HenryVarro666/siglip-embedding-tool/actions)。

## 下一步（未执行，继续不训练）

1. 人工领域审核逆变器/套管/鳍片、接线/安装、t03否定、t10导线、t12前后、e026归属、t16柜内外；修订另建数据版本并重评两模型，不回写本集。
2. 在独立开发集预定重复/停滞早拒绝、时间/长度预算、软约束或源槽位保护，再用新现场留出集测完整效果/成本；不按本16条挑规则，不放宽EOS掩盖失败。
3. 交错模型顺序/时段、多会话及CPU/GPU/内存profiling、峰值/并发/大库。当前并未实现缓存、OCR、局部特征或几何重建，也没有训练。

旧进展入口：[原始设备](equipment-benchmark-v1.md)、[中英融合](bilingual-fusion-v1.md)、[自动翻译](auto-translation-v1.md)、[更大译者](stronger-translation-v1.md)、[术语约束](terminology-constraints-v1.md)，历史台账SIG-001–019与当前SIG-020–043全部保留。上一次完整接续内容在Git提交5ef1c8f及对应报告中，不覆盖历史负结果。
