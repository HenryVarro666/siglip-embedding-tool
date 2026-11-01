# 重复检测、提前拒绝与原始OPUS回退

用户授权：继续不训练。先只用开发查询固定规则，再在同16条固定查询上比较Base/So400m与原始OPUS中英融合、现有约束失败回退。保存代码/数值/问题/改进，更新台账和接续。没有新独立现场留出、人类标签或参数微调。

## 不变项

旧translation.py、opus_constraints.py、constraint_fallback.py、fallback_benchmark.py、cli.py和两个既有冻结核心不改。新功能独立模块/API/模块CLI；原始工具默认不改。词表、源256/目标64 token、beam1/beam4、float32/MPS、EOS/完整词核验及融合0.5/0.5不变，外部解码器四文件及预训练权重不改。

## 开发与规则选择

开发集为原equipment-v1的8中文查询i01–i08，和正式t01–t16字符串分离；它们的历史结果已知，不能冒称新留出。先冻结开发字符串、候选网格和选择准则，再运行仅OPUS的观测生成（不加载SigLIP，不检索、不读取正式16的输出来选规则）。保存每4步活动beam token快照及原样最终输出/轨迹，不只是对最终译文做事后检查。

候选共同规定：decoder prompt长度1、4beam，每4新增token检查、只在min_new_tokens至60检查（64上限不算提前），对去掉开头prompt的每条beam做连续后缀周期检测；任意长度1..max_ngram_size的非特殊token块连续repeat_count次算重复。候选网格：max_ngram_size∈{8,12}、repeat_count∈{3,4}、min_new_tokens∈{24,32}、min_repeating_beams∈{3,4}，共16个。EOS/pad/start等特殊token不能单独或混在周期中触发。正当列表/重复词仍可能误拒绝，必须报告。

选择准则顺序：只保留不提前拒绝任何旧开发基线成功输出的候选；最大化被提前截停的旧开发基线失败数量；优先更高beam共识，再最小化这组失败的平均触发token；再优先更多重复次数、更小max_ngram、更大min_new_tokens及字典序ID。若无候选能停旧失败，不伪造改进，冻结最保守4beam/4repeat/max8/min32并报告未捕捉。开发集的“成功”仅指旧EOS/覆盖机械检查，不是语义正确。

冻结selected profile及开发输出/选择表后，才运行正式16。历史正式结果已知，但不得用ID或正式译文选择/调整规则；不因正式效果或漏检重调阈值。

## 运行时行为

新GuardedOPUSConstraintTranslator(base, *, profile:dict, observe_only=False, decoder_dir=...)继承旧包装器，通过已核验本地decoder callable传入每请求新建的StoppingCriteria。只在实际词汇约束路径启用；无目标词继续原有普通beam4。观测模式返回False并存检查快照；主动模式达到阈值时从停止回调抛专门ValueError中断真实生成，避免finalize给截断片段追加EOS后被误认为完整译文。

重复拒绝保存四beam原始序列、触发步/周期证据、首beam诊断英文和生成token数；last_trace.sequence/output仅作被拒绝的诊断，明确不是最终最优译文；eos_terminated=False、status=failed并含repetition_stop.triggered=True。旧fallback引擎因此真实重新执行原始beam1，再融合检索。不能返回截断片段、手工补词、放宽EOS或隐藏级联。正常成功、64上限失败、输入/基础设施错误沿用原有区分，旧生成配置/身份和原始参数不被修改。实例沿用现有单请求串行使用约束。

beam共识并不保证所有已完成候选也无效；启发式可能提前放弃旧路径可接受的输出。这不是需要隐藏的错误：正式集单列早拒绝旧接受、捕捉旧失败、漏掉旧失败的IDs/率、生成token分布和检索影响。

## 新评估

只比较三个融合入口：original_fusion（新生成原始OPUS）、fallback_opus_fusion（原64cap约束→新生成原始OPUS）、early_fallback_fusion（冻结重复规则约束→新生成原始OPUS）。3预热、16评分、5×16热请求/策略/模型，按意图/轮旋转顺序；两个模型单独进程Base→So，无其他模型作业。共480热请求、96评分尝试。当前对照全新计时，不重用旧时延。

独立同步总墙钟包含检测/sync、实际生成到中止、拒绝处理、备用翻译和检索；诊断快照和反复CPU/GPU交互的开销不能隐去。每请求保存状态/触发原因/原始轨迹、完整评分排名和逐热计时/可重建摘要。失败按零留16分母；分开主拒绝、早拒绝、回退触发、备用成功条件率和最终失败；另报仅旧六失败/实际触发子集的阶段耗时，不能相加分位数。

非早拒绝的新primary需与现有primary轨迹/输出一致（剥离新增guard诊断）；触发后最终英文/排名/备用轨迹与同次原始控制一致；允许正式集出现旧可接受→早拒绝但必须计数，不把它自动当语义错误。重复请求与跨模型的译文/全部轨迹/分支应一致，排名无漂移。冻结数据/支持文件/索引/全部模型字节SHA、开发协议/输出与profile、新代码及运行环境，比较/导出前复核。

## 交付与边界

源码新模块、独立CPU回归、开发选择证据、冻结协议、3策略真实Base/So结果、独立算法复算及复现命令；README、docs索引/报告/improvements/issue-log/continuation与AGENTS同步。保留旧核心/报告/负结果，不能将可返回或较短时间当关系语义已修复。最后验证本地、真实MPS模块CLI、wheel/独立检出与私有仓库CI。
