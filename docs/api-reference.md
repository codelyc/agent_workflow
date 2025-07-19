# API 参考文档

本文档提供AI Agents Workflow框架的完整API参考，包括所有核心类、方法和接口的详细说明。

## 目录

- [核心配置 API](#核心配置-api)
- [代理系统 API](#代理系统-api)
- [输出控制 API](#输出控制-api)
- [任务类型 API](#任务类型-api)
- [工具系统 API](#工具系统-api)
- [工作流引擎 API](#工作流引擎-api)
- [集成接口 API](#集成接口-api)

## 核心配置 API

### ConfigManager

配置管理器，负责加载和管理系统配置。

```python
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录路径
        """
    
    def load_config(self, config_name: str) -> AgentConfig:
        """加载配置文件
        
        Args:
            config_name: 配置文件名称
            
        Returns:
            AgentConfig: 代理配置对象
            
        Raises:
            ConfigNotFoundError: 配置文件不存在
            ConfigParseError: 配置文件解析错误
        """
    
    def save_config(self, config: AgentConfig, config_name: str) -> None:
        """保存配置文件
        
        Args:
            config: 代理配置对象
            config_name: 配置文件名称
        """
    
    def validate_config(self, config: AgentConfig) -> bool:
        """验证配置有效性
        
        Args:
            config: 代理配置对象
            
        Returns:
            bool: 配置是否有效
        """
    
    def get_default_config(self, agent_type: str) -> AgentConfig:
        """获取默认配置
        
        Args:
            agent_type: 代理类型
            
        Returns:
            AgentConfig: 默认配置对象
        """
```

### AgentConfig

代理配置类，定义代理的各种配置参数。

```python
@dataclass
class AgentConfig:
    """代理配置类"""
    
    # 基础配置
    agent_type: str                          # 代理类型
    llm_provider: str                        # LLM提供商
    model: str                               # 模型名称
    temperature: float = 0.7                 # 生成温度
    max_tokens: int = 2048                   # 最大token数
    timeout: int = 30                        # 超时时间
    
    # 输出控制配置
    quality_threshold: float = 0.8           # 质量阈值
    max_retries: int = 3                     # 最大重试次数
    validation_enabled: bool = True          # 启用验证
    
    # 性能配置
    enable_caching: bool = True              # 启用缓存
    cache_ttl: int = 3600                    # 缓存过期时间
    concurrent_tasks: int = 5                # 并发任务数
    
    # 自定义配置
    custom_settings: Dict[str, Any] = None   # 自定义设置
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
    
    def update(self, **kwargs) -> None:
        """更新配置
        
        Args:
            **kwargs: 要更新的配置项
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            AgentConfig: 配置对象
        """
```

### LLMManager

LLM管理器，负责管理不同的LLM提供商和客户端。

```python
class LLMManager:
    """LLM管理器"""
    
    def __init__(self):
        """初始化LLM管理器"""
    
    def get_client(self, provider: str, model: str, **kwargs) -> LLMClient:
        """获取LLM客户端
        
        Args:
            provider: LLM提供商名称
            model: 模型名称
            **kwargs: 额外配置参数
            
        Returns:
            LLMClient: LLM客户端实例
            
        Raises:
            UnsupportedProviderError: 不支持的提供商
            ModelNotFoundError: 模型不存在
        """
    
    def register_provider(self, provider_name: str, client_class: type) -> None:
        """注册LLM提供商
        
        Args:
            provider_name: 提供商名称
            client_class: 客户端类
        """
    
    def list_providers(self) -> List[str]:
        """列出所有支持的提供商
        
        Returns:
            List[str]: 提供商名称列表
        """
    
    def list_models(self, provider: str) -> List[str]:
        """列出提供商支持的模型
        
        Args:
            provider: 提供商名称
            
        Returns:
            List[str]: 模型名称列表
        """
```

## 代理系统 API

### MicroAgent

基础微代理类，所有具体代理的基类。

```python
class MicroAgent(ABC):
    """基础微代理类"""
    
    def __init__(self, config: AgentConfig):
        """初始化代理
        
        Args:
            config: 代理配置
        """
        self.config = config
        self.llm_client = self._create_llm_client()
        self.output_controller = self._create_output_controller()
        self.logger = self._create_logger()
    
    @abstractmethod
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """处理任务（抽象方法）
        
        Args:
            task: 任务定义
            
        Returns:
            Dict[str, Any]: 处理结果
        """
    
    async def validate_input(self, task: TaskDefinition) -> bool:
        """验证输入
        
        Args:
            task: 任务定义
            
        Returns:
            bool: 输入是否有效
        """
    
    async def prepare_context(self, task: TaskDefinition) -> Dict[str, Any]:
        """准备执行上下文
        
        Args:
            task: 任务定义
            
        Returns:
            Dict[str, Any]: 执行上下文
        """
    
    async def post_process(self, result: Dict[str, Any], task: TaskDefinition) -> Dict[str, Any]:
        """后处理结果
        
        Args:
            result: 原始结果
            task: 任务定义
            
        Returns:
            Dict[str, Any]: 处理后的结果
        """
    
    def get_capabilities(self) -> List[str]:
        """获取代理能力列表
        
        Returns:
            List[str]: 能力列表
        """
    
    def get_status(self) -> Dict[str, Any]:
        """获取代理状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
```

### AgentFactory

代理工厂类，负责创建和管理代理实例。

```python
class AgentFactory:
    """代理工厂类"""
    
    _agent_types: Dict[str, type] = {}
    
    @classmethod
    def create_agent(cls, agent_type: str, config: AgentConfig) -> MicroAgent:
        """创建代理实例
        
        Args:
            agent_type: 代理类型
            config: 代理配置
            
        Returns:
            MicroAgent: 代理实例
            
        Raises:
            UnknownAgentTypeError: 未知的代理类型
        """
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: type) -> None:
        """注册代理类型
        
        Args:
            agent_type: 代理类型名称
            agent_class: 代理类
        """
    
    @classmethod
    def list_agent_types(cls) -> List[str]:
        """列出所有注册的代理类型
        
        Returns:
            List[str]: 代理类型列表
        """
    
    @classmethod
    def get_agent_info(cls, agent_type: str) -> Dict[str, Any]:
        """获取代理类型信息
        
        Args:
            agent_type: 代理类型
            
        Returns:
            Dict[str, Any]: 代理信息
        """
```

### SupervisorAgent

监督代理类，负责协调和管理多个微代理。

```python
class SupervisorAgent:
    """监督代理类"""
    
    def __init__(self, config: AgentConfig):
        """初始化监督代理
        
        Args:
            config: 代理配置
        """
    
    async def coordinate_agents(self, agents: List[MicroAgent], 
                              tasks: List[TaskDefinition]) -> List[Dict[str, Any]]:
        """协调多个代理执行任务
        
        Args:
            agents: 代理列表
            tasks: 任务列表
            
        Returns:
            List[Dict[str, Any]]: 执行结果列表
        """
    
    async def distribute_tasks(self, tasks: List[TaskDefinition]) -> Dict[str, List[TaskDefinition]]:
        """分发任务给合适的代理
        
        Args:
            tasks: 任务列表
            
        Returns:
            Dict[str, List[TaskDefinition]]: 代理类型到任务列表的映射
        """
    
    async def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个代理的结果
        
        Args:
            results: 结果列表
            
        Returns:
            Dict[str, Any]: 聚合结果
        """
    
    async def monitor_performance(self) -> Dict[str, Any]:
        """监控代理性能
        
        Returns:
            Dict[str, Any]: 性能指标
        """
```

## 输出控制 API

### OutputController

输出控制器，负责管理和验证代理输出。

```python
class OutputController:
    """输出控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化输出控制器
        
        Args:
            config: 控制器配置
        """
    
    async def validate_output(self, output: Dict[str, Any], 
                            constraints: Dict[str, Any]) -> ValidationResult:
        """验证输出
        
        Args:
            output: 输出内容
            constraints: 约束条件
            
        Returns:
            ValidationResult: 验证结果
        """
    
    async def assess_quality(self, output: Dict[str, Any], 
                           criteria: Dict[str, Any]) -> QualityMetrics:
        """评估输出质量
        
        Args:
            output: 输出内容
            criteria: 评估标准
            
        Returns:
            QualityMetrics: 质量指标
        """
    
    async def format_output(self, output: Dict[str, Any], 
                          format_spec: Dict[str, Any]) -> Dict[str, Any]:
        """格式化输出
        
        Args:
            output: 原始输出
            format_spec: 格式规范
            
        Returns:
            Dict[str, Any]: 格式化后的输出
        """
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """添加验证规则
        
        Args:
            rule: 验证规则
        """
    
    def remove_validation_rule(self, rule_name: str) -> None:
        """移除验证规则
        
        Args:
            rule_name: 规则名称
        """
```

### QualityMetrics

质量指标类，定义输出质量的各种指标。

```python
@dataclass
class QualityMetrics:
    """质量指标类"""
    
    overall_score: float                     # 总体评分 (0-1)
    relevance_score: float                   # 相关性评分
    accuracy_score: float                    # 准确性评分
    completeness_score: float               # 完整性评分
    clarity_score: float                     # 清晰度评分
    
    # 具体指标
    word_count: int                          # 字数
    sentence_count: int                      # 句子数
    paragraph_count: int                     # 段落数
    
    # 质量标志
    meets_requirements: bool                 # 是否满足要求
    has_errors: bool                         # 是否有错误
    needs_improvement: bool                  # 是否需要改进
    
    # 详细信息
    strengths: List[str]                     # 优点
    weaknesses: List[str]                    # 缺点
    suggestions: List[str]                   # 改进建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 指标字典
        """
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """判断质量是否可接受
        
        Args:
            threshold: 质量阈值
            
        Returns:
            bool: 是否可接受
        """
```

### ValidationResult

验证结果类，包含验证的详细信息。

```python
@dataclass
class ValidationResult:
    """验证结果类"""
    
    is_valid: bool                           # 是否有效
    score: float                             # 验证评分
    
    # 验证详情
    passed_rules: List[str]                  # 通过的规则
    failed_rules: List[str]                  # 失败的规则
    warnings: List[str]                      # 警告信息
    errors: List[str]                        # 错误信息
    
    # 修复建议
    fix_suggestions: List[str]               # 修复建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 结果字典
        """
    
    def get_error_summary(self) -> str:
        """获取错误摘要
        
        Returns:
            str: 错误摘要
        """
```

## 任务类型 API

### TaskDefinition

任务定义类，描述要执行的任务。

```python
@dataclass
class TaskDefinition:
    """任务定义类"""
    
    task_type: str                           # 任务类型
    task_id: Optional[str] = None            # 任务ID
    priority: int = 1                        # 优先级 (1-10)
    
    # 任务内容
    prompt: str = ""                         # 任务提示
    description: str = ""                    # 任务描述
    parameters: Dict[str, Any] = None        # 任务参数
    
    # 约束条件
    constraints: Dict[str, Any] = None       # 约束条件
    requirements: List[str] = None           # 需求列表
    
    # 上下文信息
    context: Dict[str, Any] = None           # 上下文
    metadata: Dict[str, Any] = None          # 元数据
    
    # 时间信息
    created_at: datetime = None              # 创建时间
    deadline: Optional[datetime] = None      # 截止时间
    
    def __post_init__(self):
        """初始化后处理"""
        if self.task_id is None:
            self.task_id = self._generate_task_id()
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def get_parameter(self, key: str, default=None) -> Any:
        """获取任务参数
        
        Args:
            key: 参数键
            default: 默认值
            
        Returns:
            Any: 参数值
        """
    
    def get_constraint(self, key: str, default=None) -> Any:
        """获取约束条件
        
        Args:
            key: 约束键
            default: 默认值
            
        Returns:
            Any: 约束值
        """
    
    def add_requirement(self, requirement: str) -> None:
        """添加需求
        
        Args:
            requirement: 需求描述
        """
    
    def is_expired(self) -> bool:
        """检查任务是否过期
        
        Returns:
            bool: 是否过期
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 任务字典
        """
```

### TaskResult

任务结果类，包含任务执行的结果。

```python
@dataclass
class TaskResult:
    """任务结果类"""
    
    task_id: str                             # 任务ID
    status: str                              # 执行状态
    
    # 结果内容
    output: Dict[str, Any]                   # 输出内容
    quality_metrics: QualityMetrics          # 质量指标
    validation_result: ValidationResult      # 验证结果
    
    # 执行信息
    execution_time: float                    # 执行时间(秒)
    tokens_used: int                         # 使用的token数
    cost: float                              # 执行成本
    
    # 时间戳
    started_at: datetime                     # 开始时间
    completed_at: datetime                   # 完成时间
    
    # 错误信息
    error: Optional[str] = None              # 错误信息
    traceback: Optional[str] = None          # 错误堆栈
    
    def is_successful(self) -> bool:
        """检查任务是否成功
        
        Returns:
            bool: 是否成功
        """
    
    def get_output_value(self, key: str, default=None) -> Any:
        """获取输出值
        
        Args:
            key: 输出键
            default: 默认值
            
        Returns:
            Any: 输出值
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 结果字典
        """
```

## 工具系统 API

### BaseTool

基础工具类，所有工具的基类。

```python
class BaseTool(ABC):
    """基础工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化工具
        
        Args:
            config: 工具配置
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具（抽象方法）
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            Any: 执行结果
        """
    
    def get_description(self) -> str:
        """获取工具描述
        
        Returns:
            str: 工具描述
        """
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取工具参数定义
        
        Returns:
            Dict[str, Any]: 参数定义
        """
    
    def validate_parameters(self, **kwargs) -> bool:
        """验证参数
        
        Args:
            **kwargs: 参数
            
        Returns:
            bool: 参数是否有效
        """
```

### ToolManager

工具管理器，负责管理和调用工具。

```python
class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        """初始化工具管理器"""
        self._tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册工具
        
        Args:
            tool: 工具实例
        """
    
    def unregister_tool(self, tool_name: str) -> None:
        """注销工具
        
        Args:
            tool_name: 工具名称
        """
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Any: 执行结果
            
        Raises:
            ToolNotFoundError: 工具不存在
            ToolExecutionError: 工具执行错误
        """
    
    def list_tools(self) -> List[str]:
        """列出所有工具
        
        Returns:
            List[str]: 工具名称列表
        """
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """获取工具信息
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Dict[str, Any]: 工具信息
        """
```

## 工作流引擎 API

### WorkflowEngine

工作流引擎，负责执行复杂的工作流。

```python
class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化工作流引擎
        
        Args:
            config: 引擎配置
        """
    
    async def execute_workflow(self, workflow: WorkflowDefinition) -> WorkflowResult:
        """执行工作流
        
        Args:
            workflow: 工作流定义
            
        Returns:
            WorkflowResult: 执行结果
        """
    
    async def pause_workflow(self, workflow_id: str) -> None:
        """暂停工作流
        
        Args:
            workflow_id: 工作流ID
        """
    
    async def resume_workflow(self, workflow_id: str) -> None:
        """恢复工作流
        
        Args:
            workflow_id: 工作流ID
        """
    
    async def cancel_workflow(self, workflow_id: str) -> None:
        """取消工作流
        
        Args:
            workflow_id: 工作流ID
        """
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            Dict[str, Any]: 状态信息
        """
```

### WorkflowDefinition

工作流定义类，描述工作流的结构和执行逻辑。

```python
@dataclass
class WorkflowDefinition:
    """工作流定义类"""
    
    workflow_id: str                         # 工作流ID
    name: str                                # 工作流名称
    description: str                         # 描述
    
    # 工作流结构
    steps: List[WorkflowStep]                # 工作流步骤
    dependencies: Dict[str, List[str]]       # 步骤依赖关系
    
    # 执行配置
    execution_mode: str = "sequential"       # 执行模式
    timeout: int = 3600                      # 超时时间
    retry_policy: Dict[str, Any] = None      # 重试策略
    
    # 条件控制
    conditions: Dict[str, Any] = None        # 执行条件
    
    def add_step(self, step: 'WorkflowStep') -> None:
        """添加工作流步骤
        
        Args:
            step: 工作流步骤
        """
    
    def add_dependency(self, step_id: str, depends_on: List[str]) -> None:
        """添加步骤依赖
        
        Args:
            step_id: 步骤ID
            depends_on: 依赖的步骤ID列表
        """
    
    def validate(self) -> bool:
        """验证工作流定义
        
        Returns:
            bool: 定义是否有效
        """
```

## 异常类型

框架定义的自定义异常类型。

```python
class AgentWorkflowError(Exception):
    """基础异常类"""
    pass

class ConfigError(AgentWorkflowError):
    """配置相关异常"""
    pass

class ConfigNotFoundError(ConfigError):
    """配置文件不存在"""
    pass

class ConfigParseError(ConfigError):
    """配置解析错误"""
    pass

class AgentError(AgentWorkflowError):
    """代理相关异常"""
    pass

class UnknownAgentTypeError(AgentError):
    """未知代理类型"""
    pass

class AgentExecutionError(AgentError):
    """代理执行错误"""
    pass

class LLMError(AgentWorkflowError):
    """LLM相关异常"""
    pass

class UnsupportedProviderError(LLMError):
    """不支持的LLM提供商"""
    pass

class ModelNotFoundError(LLMError):
    """模型不存在"""
    pass

class ToolError(AgentWorkflowError):
    """工具相关异常"""
    pass

class ToolNotFoundError(ToolError):
    """工具不存在"""
    pass

class ToolExecutionError(ToolError):
    """工具执行错误"""
    pass

class WorkflowError(AgentWorkflowError):
    """工作流相关异常"""
    pass

class WorkflowValidationError(WorkflowError):
    """工作流验证错误"""
    pass

class WorkflowExecutionError(WorkflowError):
    """工作流执行错误"""
    pass
```

## 使用示例

### 基础使用

```python
import asyncio
from src.core.config.config_manager import ConfigManager
from src.agents.factory.agent_factory import AgentFactory
from src.core.task_types.task_definition import TaskDefinition

async def basic_usage_example():
    # 1. 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config('code_assistant_config.yaml')
    
    # 2. 创建代理
    agent = AgentFactory.create_agent('code_assistant', config)
    
    # 3. 定义任务
    task = TaskDefinition(
        task_type='code_generation',
        prompt='创建一个排序算法',
        parameters={
            'language': 'python',
            'algorithm': 'quicksort'
        },
        constraints={
            'max_lines': 50,
            'include_comments': True
        }
    )
    
    # 4. 执行任务
    result = await agent.process_task(task)
    
    # 5. 处理结果
    print(f"生成的代码: {result['output']}")
    print(f"质量评分: {result['quality_metrics'].overall_score}")

# 运行示例
asyncio.run(basic_usage_example())
```

### 高级使用

```python
import asyncio
from src.workflows.engine.workflow_engine import WorkflowEngine
from src.workflows.templates.sequential_workflow import SequentialWorkflow
from src.agents.supervisor.supervisor_agent import SupervisorAgent

async def advanced_usage_example():
    # 1. 创建监督代理
    supervisor_config = ConfigManager().load_config('supervisor_config.yaml')
    supervisor = SupervisorAgent(supervisor_config)
    
    # 2. 创建工作流
    workflow = SequentialWorkflow([
        {
            'agent_type': 'code_assistant',
            'task': {
                'type': 'code_generation',
                'prompt': '创建一个Web API'
            }
        },
        {
            'agent_type': 'data_analysis',
            'task': {
                'type': 'performance_analysis',
                'target': 'generated_api'
            }
        },
        {
            'agent_type': 'content_generation',
            'task': {
                'type': 'documentation',
                'target': 'api_documentation'
            }
        }
    ])
    
    # 3. 执行工作流
    engine = WorkflowEngine()
    result = await engine.execute_workflow(workflow)
    
    # 4. 监控和分析
    performance = await supervisor.monitor_performance()
    print(f"工作流执行结果: {result}")
    print(f"性能指标: {performance}")

# 运行示例
asyncio.run(advanced_usage_example())
```

---

本API参考文档涵盖了框架的核心功能和接口。更多详细信息和示例，请参考相应的模块文档和源代码注释。