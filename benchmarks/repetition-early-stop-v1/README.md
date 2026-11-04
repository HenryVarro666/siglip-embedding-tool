# 重复检测与提前拒绝：开发冻结后同16查询比较

本轮不训练。比较原始OPUS中英融合、现有64-token约束失败→原始OPUS融合、重复早拒绝→原始OPUS融合。沿用同16中文意图/40图及原标签，文件逐字节不变，参考英文不进入译者。查询已有历史诊断，不是新独立留出；规则只用另外8条旧开发中文确定。[完整结果与限制](../../docs/repetition-early-stop-v1.md)。完整问题/修复见[台账SIG-044起](../../docs/issue-log.md)，后续从[接续状态](../../docs/continuation.md)进入。

## 开发经过与最终规则

- 阶段1在首次模型观测前提交a7404e2，预声明16个严格后缀候选。7条机械成功、1条i07无EOS，全部候选都没捕捉到失败，也没误拒绝成功。按原准则的保守profile不能节省时间；源码/全部输出保存在7fb0973及[development-stage1](development-stage1/README.md)。
- 仅根据开发beam的强制术语尾部插入现象，阶段2在2802893重新声明有界尾部检测和16候选，再对同8源句重新真实观测/主动验证。不是声称一开始就声明了阶段2，也没使用新正式结果选规则。
- 最终profile于905f130冻结：`bounded-n16-r2-min32-q4-tail4`。从32新增token起每4步检查，64上限不算提前；周期4–16token连续重复至少2次，允许其后最多4尾token，四条活动beam都须独立符合。特殊token不属于循环。只在实际词汇约束路径启用；不修改生成token或译文。
- 开发i07实际在48步中断，重新运行原始OPUS；7旧机械成功未提前拒绝。但这7条没有进入最终32步检测区间（或无目标词、guard关闭），对长正常句误拒绝缺少覆盖，不能据此保证安全。
- `dev-protocol/observations/selection/validation.json`、`dev-queries.json`和profile均保存；模型分配前会重放选择和语义关系，不能连改摘要后拿另一个候选冒充选中规则。

## 正式冻结与计时

正式冻结提交b6d0aec，核心SHA `5508fc7372887770f584065949d19debe77b81fd6d09cf022a432de164d7bc12`，协议SHA `cfdc70f0d5c75bb5cd2fb242c5eaed033b7b7c975f72f273028659c13ab59229`，profile字节SHA `e93ee35b4b81fe94a7b2d97c25fa3317bce599e6976d1a54a6a2d90d764b5777`。旧44b/e377核心和旧CLI不改；检索模型/索引/权重字节、OPUS/解码文件、查询/标签/支持文件、开发证据和代码均绑定，保留旧负结果。

3策略各3预热、16完整40图评分、5×16热请求，按意图/轮旋转。Base→So400m分开进程，共96评分、480热请求。每次自动请求真实生成，回退也重新生成，无译文缓存。同步独立墙钟包含检测/beam复制/异常处理/失败成本/备用翻译/检索；阶段分位数不能相加。序列化/保存不在请求内，首调用/加载单列。所有成功、失败、尾样本保留，不做事后删样。

记录主拒绝、早拒绝、回退触发、备用成功条件率、最终失败；失败指标按零留16分母。另报旧六失败、漏检、旧接受→早拒绝等同组时间与逐条影响。活动beam重复不保证没有已完成可用候选，误拒绝不得隐藏。序列长度、已观测回调步数与真实中止步数分开，正常finalize长度不是通用的decoder计算次数。

## 复现

```bash
uv sync --frozen --extra dev --extra translation
# 权重/解码器和原40图/两索引沿用上一轮；来源/下载/建库见其README。
uv run python -m siglip_tools.early_stop_benchmark run \
  --model base --index artifacts/equipment-benchmark/base-v1/index.npz \
  --output artifacts/repetition-early-stop/base-v1
uv run python -m siglip_tools.early_stop_benchmark run \
  --model quality --index artifacts/equipment-benchmark/quality-v1/index.npz \
  --output artifacts/repetition-early-stop/quality-v1
uv run python -m siglip_tools.early_stop_benchmark compare \
  --base artifacts/repetition-early-stop/base-v1/result.json \
  --quality artifacts/repetition-early-stop/quality-v1/result.json \
  --output artifacts/repetition-early-stop/comparison.json
uv run python examples/summarize_early_stop_benchmark.py \
  --base artifacts/repetition-early-stop/base-v1/result.json \
  --quality artifacts/repetition-early-stop/quality-v1/result.json \
  --output artifacts/repetition-early-stop/reexport-v1
uv run python examples/audit_early_stop_benchmark.py \
  --base artifacts/repetition-early-stop/base-v1/result.json \
  --quality artifacts/repetition-early-stop/quality-v1/result.json \
  --output artifacts/repetition-early-stop/reaudit-v1.json
```

输出存在时拒绝覆盖，重测使用新路径。图片、模型、索引和原始结果只在本机忽略目录；Git内`results-summary.json`保存完整指标/阶段/分组统计，`requests.json`保存完整评分排名、原样部分/最终轨迹和逐热计时/可重建摘要；`audit.json`与`verification.json`保存复算和验收口径。支持恢复与源字节一致的旧索引，不能绕过身份检查。下载/建库及现有模型来源见[上一轮](../constraint-fallback-v1/README.md)。

开发重跑需在新目录复制相应8源句并先声明协议，执行模块`early_stop_development observe/select/validate`；默认命令会拒绝覆盖现有开发文件。阶段1源码重现需检出7fb0973，当前阶段2代码不能冒充旧协议的执行代码。改变规则/源句/标签必须新实验，不再沿用本轮成绩。

## 直接调用

独立模块入口使用最终打包profile，旧`siglip-tools`接口保持原状：

```bash
uv run python -m siglip_tools.repetition_stop search \
  --index artifacts/equipment-benchmark/base-v1/index.npz \
  --query '变压器的散热片在垂下的黑色粗电缆后方，电缆连接顶部的套管。' \
  --mode auto_fusion --fallback opus --device mps --local-files-only
```

JSON保留主/备用轨迹、部分英文的诊断角色、实际使用文本/路由/TopK与完整计时，失败片段不会进入embedding。预期最终失败输出JSON并退出2；基础设施错误不会变成静默回退。

Python可复用原始OPUS/GuardedOPUSConstraintTranslator、SigLIPEmbedder和ImageIndex，再调用现有`timed_fallback_search(...,mode='auto_fusion',fallback='opus')`。显式传入`json.load(open(.../repetition_profile_v1.json))`作为guard的profile；原始与guard共享同一套冻结OPUS权重。

早停是成本启发式，不证明术语/接线/否定/安装关系正确。规则已锁定，即使正式结果不好也不就地重调。

## 冻结源码快照（2026-11-04 起）

`frozen-source.json` 记录测量提交 `f079027` 中被实现指纹绑定的源码文件 SHA-256。冻结记录的 `implementation_sha256` 按该快照复算，工作区源码可以维护而不改写测量；执行本协议的新测量仍要求当前源码与快照一致，否则属于新版本。核对：`uv run python -m siglip_tools.frozen_source verify benchmarks/repetition-early-stop-v1`。
