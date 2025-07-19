# AI Agents Workflow

一个基于Python的智能代理工作流框架，支持多种类型的AI代理协作完成复杂任务。

## 项目概述

本项目提供了一个完整的AI代理工作流系统，包含以下核心功能：

- **多类型代理支持**：代码助手、内容生成、数据分析等专业代理
- **统一配置管理**：集中化的配置系统，支持多种LLM提供商
- **输出控制系统**：智能的输出管理和质量控制
- **任务追踪**：完整的任务执行追踪和日志记录
- **工具集成**：丰富的工具库支持各种操作

## 项目结构

```
ai-agents-workflow/
├── src/                    # 核心源代码
│   ├── core/              # 核心模块
│   │   ├── config/        # 配置管理
│   │   ├── output_control/ # 输出控制
│   │   ├── memory/        # 内存管理
│   │   ├── task_types/    # 任务类型定义
│   │   └── tracing/       # 追踪系统
│   ├── agents/            # 代理实现
│   │   ├── factory/       # 代理工厂
│   │   ├── micro/         # 微代理
│   │   └── supervisor/    # 监督代理
│   ├── tools/             # 工具库
│   ├── workflows/         # 工作流引擎
│   └── integrations/      # 第三方集成
├── AgentWorker/           # 代理工作器示例
│   ├── CodeAssistant/     # 代码助手代理
│   ├── ContentGeneration/ # 内容生成代理
│   └── DataAnalysis/      # 数据分析代理
├── docs/                  # 文档目录
└── logs/                  # 日志文件
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd agent_workflow

# 安装uv包管理器（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或在Windows上：powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 使用uv安装依赖
uv sync

# 传统方式（可选）
# pip install -e .
```

### 2. 配置设置

在运行代理之前，需要配置LLM提供商的API密钥：

```python
# 示例：配置OpenAI
export OPENAI_API_KEY="your-api-key"

# 或在代码中配置
from src.core.config.llm_manager import LLMManager
llm_manager = LLMManager()
```

### 3. 运行示例代理

```bash
# 运行代码助手代理
python -m AgentWorker.CodeAssistant.code_assistant_worker

# 运行内容生成代理
python -m AgentWorker.ContentGeneration.content_generation_worker

# 运行数据分析代理
python -m AgentWorker.DataAnalysis.data_analysis_worker
```

## 核心组件

### 配置管理 (Core Config)

- **ConfigManager**: 统一配置管理器
- **LLMManager**: LLM提供商管理
- **AgentConfig**: 代理配置定义

### 输出控制 (Output Control)

- **OutputController**: 输出质量控制
- **QualityMetrics**: 质量评估指标
- **ValidationRules**: 验证规则引擎

### 代理系统 (Agents)

- **MicroAgent**: 基础微代理
- **SupervisorAgent**: 监督代理
- **AgentFactory**: 代理工厂模式

### 工具库 (Tools)

- **分析工具**: 数据分析、统计计算
- **开发工具**: 代码生成、测试、调试
- **外部工具**: API调用、文件操作
- **操作工具**: 系统操作、流程控制

## 使用示例

### 创建自定义代理

```python
from src.agents.micro.base_agent import MicroAgent
from src.core.config.agent_config import AgentConfig
from src.core.task_types.task_definition import TaskDefinition

class CustomAgent(MicroAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    async def process_task(self, task: TaskDefinition) -> dict:
        # 实现自定义任务处理逻辑
        result = await self.llm_client.generate(
            prompt=task.prompt,
            constraints=task.constraints
        )
        return result

# 使用代理
config = AgentConfig(
    agent_type="custom",
    llm_provider="openai",
    model="gpt-4"
)
agent = CustomAgent(config)
```

### 配置工作流

```python
from src.workflows.engine.workflow_engine import WorkflowEngine
from src.workflows.templates.sequential_workflow import SequentialWorkflow

# 创建工作流
workflow = SequentialWorkflow([
    {"agent": "code_assistant", "task": "generate_function"},
    {"agent": "data_analysis", "task": "analyze_performance"},
    {"agent": "content_generation", "task": "create_documentation"}
])

# 执行工作流
engine = WorkflowEngine()
result = await engine.execute(workflow)
```

## 配置选项

### LLM提供商配置

支持多种LLM提供商：

- **OpenAI**: GPT-3.5, GPT-4系列
- **Anthropic**: Claude系列
- **Google**: Gemini系列
- **本地模型**: Ollama, LM Studio等

### 代理配置

```yaml
# agent_config.yaml
agent:
  type: "code_assistant"
  llm_provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2048
  
output_control:
  quality_threshold: 0.8
  max_retries: 3
  validation_enabled: true
  
logging:
  level: "INFO"
  file_path: "logs/agent.log"
```

## 开发指南

### 添加新的代理类型

1. 在 `src/agents/micro/` 下创建新的代理类
2. 继承 `MicroAgent` 基类
3. 实现 `process_task` 方法
4. 在 `AgentFactory` 中注册新代理

### 添加新的工具

1. 在 `src/tools/` 相应分类下创建工具类
2. 实现工具接口
3. 在 `tool_descriptions.py` 中添加描述
4. 更新工具注册表

### 自定义输出控制

1. 继承 `OutputController` 类
2. 实现自定义验证逻辑
3. 配置质量评估指标
4. 集成到代理配置中

## 故障排除

### 常见问题

1. **模块导入错误**
   - 确保项目已正确安装：`uv sync` 或 `pip install -e .`
   - 检查Python路径配置

2. **API密钥错误**
   - 验证环境变量设置
   - 检查API密钥有效性

3. **代理执行失败**
   - 查看日志文件：`logs/`
   - 检查配置文件格式
   - 验证任务定义

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用追踪
from src.core.tracing import enable_tracing
enable_tracing()
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

更多详细文档请参考 `docs/` 目录下的其他文件。