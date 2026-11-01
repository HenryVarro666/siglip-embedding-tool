# 约束失败后回退：设计与固定口径

执行用户授权的预训练推理比较，不训练。沿用terminology-constraints-v1的16中文意图、40图库、配对英文参考及全部标签/备注，逐字节相同。此集已有诊断结果，不能称新独立留出集。不调整词表、beam、64-token预算、融合0.5/0.5或标签。

新增timed_fallback_search(index, encoder, constrained, original, chinese, mode='auto_en', fallback='error', top_k=5)。返回ok、translation、matches、failure、primary_failure、constraint_trace、fallback_trace、fallback_triggered、effective_route、query_texts、timing。两译者共享冻结OPUS权重。约束成功不调用备用译者；只有本次生成输出被拒绝（source匹配、failed trace且有sequence/output）触发回退。输入/来源/程序/系统错误不包装成成功；检索错误不触发翻译回退。

- error：保持失败，不检索被拒绝英文。
- opus：本请求重新调用原始beam1 OPUS，再按英文/中英融合检索；备用翻译失败则最终失败，不级联中文。
- chinese：失败后仅对中文检索一次；即使请求融合，也不重复编码中文，不使用失败英文。

总耗时用同步设备后的独立墙钟，从约束尝试开始到最终检索结束，包含失败生成/校验、异常处理、备用翻译和检索。同步阶段单列constraint_translation_ms、fallback_translation_ms、retrieval_ms；诊断序列化放在计时后。首请求/模型加载单列。无译文缓存。失败文本/序列、原因与最终使用文本分别保存。

9策略：zh、original_en、original_fusion、constrained_en、constrained_fusion、fallback_opus_en、fallback_opus_fusion、fallback_zh_en、fallback_zh_fusion。同期控制全部重新实测。每模型每策略3预热、16完整评分、5轮×16意图热请求，按意图/轮旋转策略顺序。Base、So400m顺序执行，单进程单检索模型。不复制旧耗时或通过旧结果路由；本轮不重测普通beam4。

逐评分尝试保存完整40图排序/分数和两阶段轨迹；每热请求保存独立计时、失败/分支状态及输出/轨迹/排序摘要。断言同策略重复无漂移、成功约束分支与约束控制一致、OPUS回退与原始控制一致、中文回退与直接中文控制一致。跨模型核对译文/轨迹/分支。失败按零留在16意图主分母，分别报primary rejection、fallback trigger、fallback success条件分母及terminal failure，不能把可返回结果当语义正确。

旧测量核心44b19b…及导出器不动。新模块/CLI/API纳入新实现摘要；模型评分前冻结数据、代码、来源、模型权重字节SHA和索引。独立输出目录保留历史负结果。CLI新增--constraint-fallback error|opus|chinese，非error只适用于OPUS constrained自动翻译，无效组合在模型分配前拒绝。

实施者决定：5轮热重复（80样本/策略），相对旧10轮属于新协议，不声称同次测量。缓存/并发、人类领域审核、现场新数据及语义关系改进未执行。持续更新issue-log、improvements、文档索引与continuation。
