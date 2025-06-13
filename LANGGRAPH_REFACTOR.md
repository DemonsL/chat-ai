# LangGraph 多轮对话重构文档

## 概述

本次重构将原有的 LLM 管理系统迁移到基于 LangGraph 的状态图架构，实现更强大的多轮对话管理和状态跟踪功能。

## 架构变化

### 1. 核心组件重构

#### 原架构
```python
# 原有的 LLMManager
class LLMManager:
    async def process_chat(...)
    async def process_rag(...)
    async def process_agent(...)
```

#### 新架构
```python
# 基于 LangGraph 的 LLMManager
class LLMManager:
    async def process_conversation(...)  # 统一的对话处理接口
    def _build_chat_graph(...)           # 聊天模式状态图
    def _build_rag_graph(...)            # RAG模式状态图
    def _build_agent_graph(...)          # Agent模式状态图
```

### 2. 状态管理

#### ConversationState 状态定义
```python
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    model_config: Dict[str, Any]
    system_prompt: Optional[str]
    mode: str
    retrieved_documents: Optional[List[str]]
    available_tools: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    user_query: Optional[str]
    final_response: Optional[str]
```

### 3. 状态图构建

#### 聊天模式图
```
START → chat_node → END
```

#### RAG模式图
```
START → prepare_context → rag_response → END
```

#### Agent模式图
```
START → agent_planning → agent_response → END
```

## 主要特性

### 1. 统一的对话接口
- 所有模式（chat/rag/agent）通过 `process_conversation` 统一处理
- 自动的状态管理和消息累积
- 支持流式响应

### 2. 智能模式选择
- 自动根据上下文选择合适的处理模式
- 支持意图识别和动态路由
- 兼容现有的元数据配置

### 3. 缓存机制
- 模型实例缓存
- 状态图缓存
- 提高性能和减少初始化开销

### 4. 向后兼容
- 保留原有的 `process_chat`、`process_rag`、`process_agent` 方法
- 现有调用代码无需修改

## 使用示例

### 1. 基础聊天
```python
llm_manager = LLMManager()

async for chunk in llm_manager.process_conversation(
    messages=[{"role": "user", "content": "你好"}],
    model_config=model_config,
    mode="chat"
):
    chunk_data = json.loads(chunk)
    print(chunk_data.get("content", ""), end="")
```

### 2. RAG 检索增强
```python
async for chunk in llm_manager.process_conversation(
    messages=messages,
    model_config=model_config,
    mode="rag",
    retrieved_documents=["文档1", "文档2"]
):
    chunk_data = json.loads(chunk)
    print(chunk_data.get("content", ""), end="")
```

### 3. Agent 工具使用
```python
async for chunk in llm_manager.process_conversation(
    messages=messages,
    model_config=model_config,
    mode="agent",
    available_tools=["search", "analysis"]
):
    chunk_data = json.loads(chunk)
    print(chunk_data.get("content", ""), end="")
```

### 4. 多轮对话
```python
conversation_history = []

for user_input in ["消息1", "消息2", "消息3"]:
    conversation_history.append({"role": "user", "content": user_input})
    
    assistant_response = ""
    async for chunk in llm_manager.process_conversation(
        messages=conversation_history.copy(),
        model_config=model_config,
        mode="chat"
    ):
        chunk_data = json.loads(chunk)
        content = chunk_data.get("content", "")
        assistant_response += content
    
    conversation_history.append({"role": "assistant", "content": assistant_response})
```

## MessageService 集成

### 重构要点
1. 移除了复杂的模式判断逻辑
2. 统一使用 `process_conversation` 方法
3. 保持所有现有功能（意图识别、文档检索等）
4. 改进了错误处理和元数据管理

### 关键变化
```python
# 原有的复杂逻辑
if processing_mode == "chat":
    service_stream = self.llm_orchestrator.process_chat(...)
elif processing_mode == "rag":
    service_stream = self.llm_orchestrator.process_rag(...)
elif processing_mode == "agent":
    service_stream = self.llm_orchestrator.process_agent(...)

# 新的统一接口
service_stream = self.llm_orchestrator.process_conversation(
    messages=messages,
    model_config=model_config_dict,
    mode=processing_mode,
    system_prompt=conversation.system_prompt,
    retrieved_documents=retrieved_docs if processing_mode == "rag" else None,
    available_tools=available_tools if processing_mode == "agent" else None,
    metadata=processing_metadata,
)
```

## 性能优化

### 1. 图缓存
- 每种模式的状态图只构建一次
- 减少重复的图编译开销
- 支持动态缓存清理

### 2. 模型缓存
- 相同配置的模型实例复用
- 避免重复的模型初始化
- 内存使用优化

### 3. 状态管理
- 轻量级状态传递
- 最小化数据复制
- 高效的消息累积

## 扩展性

### 1. 新增处理模式
```python
def _build_custom_graph(self) -> StateGraph:
    """构建自定义模式的状态图"""
    def custom_node(state: ConversationState) -> Dict[str, Any]:
        # 自定义处理逻辑
        pass
    
    graph_builder = StateGraph(ConversationState)
    graph_builder.add_node("custom", custom_node)
    graph_builder.add_edge(START, "custom")
    graph_builder.add_edge("custom", END)
    
    return graph_builder.compile()
```

### 2. 复杂工作流
```python
def _build_complex_graph(self) -> StateGraph:
    """构建复杂的多步骤工作流"""
    graph_builder = StateGraph(ConversationState)
    
    # 添加多个节点
    graph_builder.add_node("step1", step1_node)
    graph_builder.add_node("step2", step2_node)
    graph_builder.add_node("step3", step3_node)
    
    # 定义流程
    graph_builder.add_edge(START, "step1")
    graph_builder.add_edge("step1", "step2")
    graph_builder.add_edge("step2", "step3")
    graph_builder.add_edge("step3", END)
    
    return graph_builder.compile()
```

### 3. 条件分支
```python
def route_function(state: ConversationState) -> str:
    """根据状态决定下一个节点"""
    if state.get("needs_retrieval"):
        return "rag_node"
    else:
        return "chat_node"

graph_builder.add_conditional_edges(
    "decision_node",
    route_function,
    {
        "rag_node": "rag_node",
        "chat_node": "chat_node"
    }
)
```

## 部署和监控

### 1. 日志记录
- 图执行的详细日志
- 状态转换跟踪
- 性能指标收集

### 2. 错误处理
- 节点级别的错误捕获
- 优雅的降级策略
- 详细的错误信息

### 3. 监控指标
- 图执行时间
- 缓存命中率
- 内存使用情况

## 迁移指南

### 1. 无缝升级
- 现有代码无需修改
- 向后兼容的API
- 渐进式迁移

### 2. 新功能采用
```python
# 从这样的调用
async for chunk in llm_manager.process_chat(messages, config):
    pass

# 迁移到这样的调用
async for chunk in llm_manager.process_conversation(
    messages=messages, 
    model_config=config, 
    mode="chat"
):
    pass
```

### 3. 测试验证
- 运行 `demo_langgraph.py` 验证功能
- 对比新旧实现的输出
- 性能基准测试

## 依赖要求

```bash
pip install langgraph
pip install langchain
pip install typing-extensions
```

## 总结

本次重构实现了：
1. ✅ 基于 LangGraph 的状态图架构
2. ✅ 统一的对话处理接口
3. ✅ 智能的状态管理
4. ✅ 高效的缓存机制
5. ✅ 完整的向后兼容性
6. ✅ 可扩展的架构设计

通过 LangGraph 的强大功能，系统现在具备了更好的多轮对话管理能力、状态跟踪功能和扩展性，为未来的功能增强奠定了坚实的基础。 