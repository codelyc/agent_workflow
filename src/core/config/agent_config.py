"""
智能体配置数据类

定义智能体、工具、模型等配置的数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentType(Enum):
    """智能体类型"""
    SUPERVISOR = "supervisor"
    MICRO = "micro"


class TaskCategory(Enum):
    """任务类别"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    OPTIMIZATION = "optimization"
    SEARCH = "search"
    EXECUTION = "execution"
    VALIDATION = "validation"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        """配置验证"""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass 
class ToolConfig:
    """工具配置"""
    name: str
    module: str
    function: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """配置验证"""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.module or not self.function:
            raise ValueError("Tool module and function must be specified")


@dataclass
class MemoryConfig:
    """记忆配置"""
    enabled: bool = True
    storage_path: Optional[str] = None
    cache_size: int = 1000
    cleanup_interval: int = 3600  # seconds
    
    def __post_init__(self):
        """配置验证"""
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")


@dataclass
class TracingConfig:
    """追踪配置"""
    enabled: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_private_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        """配置验证"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")


@dataclass
class AgentConfig:
    """智能体配置"""
    name: str
    type: AgentType
    description: str = ""
    
    # 任务配置
    task_categories: List[TaskCategory] = field(default_factory=list)
    default_task_type: Optional[TaskCategory] = None
    
    # 模型配置
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="gpt-4o-mini"))
    
    # 工具配置
    tools: List[ToolConfig] = field(default_factory=list)
    
    # 执行配置
    max_steps: int = 50
    timeout: int = 300  # seconds
    max_retries: int = 3
    
    # SOP配置
    sop_category: Optional[str] = None
    sop_path: Optional[str] = None
    
    # 微智能体特定配置
    micro_agents: List[str] = field(default_factory=list)  # 对于supervisor agent
    
    # 系统提示配置
    system_prompt: Optional[str] = None
    system_prompt_template: Optional[str] = None
    
    # 记忆和追踪配置
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    
    # 其他配置
    workspace_path: Optional[str] = None
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """配置后验证"""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        
        # 设置默认任务类型
        if not self.default_task_type and self.task_categories:
            self.default_task_type = self.task_categories[0]
        
        # 验证SOP配置
        if self.type == AgentType.SUPERVISOR and not self.sop_category:
            self.sop_category = "custom_workflow"
    
    def get_tool_by_name(self, tool_name: str) -> Optional[ToolConfig]:
        """根据名称获取工具配置"""
        return next((tool for tool in self.tools if tool.name == tool_name), None)
    
    def is_enabled_tool(self, tool_name: str) -> bool:
        """检查工具是否启用"""
        tool = self.get_tool_by_name(tool_name)
        return tool is not None and tool.enabled
    
    def get_enabled_tools(self) -> List[ToolConfig]:
        """获取所有启用的工具"""
        return [tool for tool in self.tools if tool.enabled]
    
    def can_handle_task(self, task_category: TaskCategory) -> bool:
        """检查是否能处理特定类型的任务"""
        return task_category in self.task_categories
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'task_categories': [cat.value for cat in self.task_categories],
            'default_task_type': self.default_task_type.value if self.default_task_type else None,
            'model': {
                'name': self.model.name,
                'max_tokens': self.model.max_tokens,
                'temperature': self.model.temperature,
                'timeout': self.model.timeout,
                'retry_attempts': self.model.retry_attempts
            },
            'tools': [
                {
                    'name': tool.name,
                    'module': tool.module,
                    'function': tool.function,
                    'description': tool.description,
                    'enabled': tool.enabled,
                    'parameters': tool.parameters
                }
                for tool in self.tools
            ],
            'max_steps': self.max_steps,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'sop_category': self.sop_category,
            'sop_path': self.sop_path,
            'micro_agents': self.micro_agents,
            'system_prompt': self.system_prompt,
            'system_prompt_template': self.system_prompt_template,
            'memory': {
                'enabled': self.memory.enabled,
                'storage_path': self.memory.storage_path,
                'cache_size': self.memory.cache_size,
                'cleanup_interval': self.memory.cleanup_interval
            },
            'tracing': {
                'enabled': self.tracing.enabled,
                'log_level': self.tracing.log_level
            },
            'workspace_path': self.workspace_path,
            'enabled': self.enabled,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    description: str = ""
    
    # 执行配置
    supervisor_agent: str = ""  # 主管智能体名称
    micro_agents: List[str] = field(default_factory=list)  # 微智能体列表
    
    # 执行策略
    execution_strategy: str = "sequential"  # sequential, parallel, conditional
    max_concurrent_tasks: int = 3
    timeout: int = 600  # seconds
    
    # 输入输出配置
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "markdown"  # markdown, json, yaml
    
    # 环境配置
    workspace_path: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # 其他配置
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """配置后验证"""
        if not self.name:
            raise ValueError("Workflow name cannot be empty")
        
        if not self.supervisor_agent:
            raise ValueError("Supervisor agent must be specified")
        
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        valid_strategies = ["sequential", "parallel", "conditional"]
        if self.execution_strategy not in valid_strategies:
            raise ValueError(f"execution_strategy must be one of {valid_strategies}")
        
        valid_formats = ["markdown", "json", "yaml", "text"]
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")


import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentConfigManager:
    """Agent配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为当前目录下的configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.agents: Dict[str, AgentConfig] = {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        self._loaded = False
    
    def load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """从文件加载配置"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif config_file.endswith('.json'):
                    import json
                    return json.load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {config_file}")
                    return {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            return {}
    
    def load_agent_config(self, agent_name: str, config_data: Dict[str, Any]) -> AgentConfig:
        """加载Agent配置"""
        try:
            # 解析模型配置
            model_data = config_data.get('model', {})
            model = ModelConfig(
                name=model_data.get('name', 'gpt-4o-mini'),
                api_key=model_data.get('api_key'),
                base_url=model_data.get('base_url'),
                max_tokens=model_data.get('max_tokens', 4000),
                temperature=model_data.get('temperature', 0.1),
                timeout=model_data.get('timeout', 30),
                retry_attempts=model_data.get('retry_attempts', 3)
            )
            
            # 解析工具配置
            tools = []
            for tool_data in config_data.get('tools', []):
                tool = ToolConfig(
                    name=tool_data['name'],
                    module=tool_data['module'],
                    function=tool_data['function'],
                    description=tool_data.get('description', ''),
                    parameters=tool_data.get('parameters', {}),
                    enabled=tool_data.get('enabled', True)
                )
                tools.append(tool)
            
            # 解析任务类别
            task_categories = []
            for cat_str in config_data.get('task_categories', []):
                try:
                    task_categories.append(TaskCategory(cat_str))
                except ValueError:
                    logger.warning(f"未知的任务类别: {cat_str}")
            
            # 解析Agent类型
            agent_type_str = config_data.get('type', 'micro')
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                logger.warning(f"未知的Agent类型: {agent_type_str}，使用默认值micro")
                agent_type = AgentType.MICRO
            
            # 解析记忆配置
            memory_data = config_data.get('memory', {})
            memory = MemoryConfig(
                enabled=memory_data.get('enabled', True),
                storage_path=memory_data.get('storage_path'),
                cache_size=memory_data.get('cache_size', 1000),
                cleanup_interval=memory_data.get('cleanup_interval', 3600)
            )
            
            # 解析追踪配置
            tracing_data = config_data.get('tracing', {})
            tracing = TracingConfig(
                enabled=tracing_data.get('enabled', True),
                langfuse_public_key=tracing_data.get('langfuse_public_key'),
                langfuse_private_key=tracing_data.get('langfuse_private_key'),
                langfuse_host=tracing_data.get('langfuse_host'),
                log_level=tracing_data.get('log_level', 'INFO')
            )
            
            # 创建Agent配置
            agent_config = AgentConfig(
                name=agent_name,
                type=agent_type,
                description=config_data.get('description', ''),
                task_categories=task_categories,
                default_task_type=TaskCategory(config_data['default_task_type']) if config_data.get('default_task_type') else None,
                model=model,
                tools=tools,
                max_steps=config_data.get('max_steps', 50),
                timeout=config_data.get('timeout', 300),
                max_retries=config_data.get('max_retries', 3),
                sop_category=config_data.get('sop_category'),
                sop_path=config_data.get('sop_path'),
                micro_agents=config_data.get('micro_agents', []),
                system_prompt=config_data.get('system_prompt'),
                system_prompt_template=config_data.get('system_prompt_template'),
                memory=memory,
                tracing=tracing,
                workspace_path=config_data.get('workspace_path'),
                enabled=config_data.get('enabled', True),
                tags=config_data.get('tags', []),
                metadata=config_data.get('metadata', {})
            )
            
            return agent_config
            
        except Exception as e:
            logger.error(f"加载Agent配置失败 {agent_name}: {e}")
            raise
    
    def load_workflow_config(self, workflow_name: str, config_data: Dict[str, Any]) -> WorkflowConfig:
        """加载Workflow配置"""
        try:
            workflow_config = WorkflowConfig(
                name=workflow_name,
                description=config_data.get('description', ''),
                supervisor_agent=config_data.get('supervisor_agent', ''),
                micro_agents=config_data.get('micro_agents', []),
                execution_strategy=config_data.get('execution_strategy', 'sequential'),
                max_concurrent_tasks=config_data.get('max_concurrent_tasks', 3),
                timeout=config_data.get('timeout', 600),
                input_schema=config_data.get('input_schema', {}),
                output_format=config_data.get('output_format', 'markdown'),
                workspace_path=config_data.get('workspace_path'),
                environment_variables=config_data.get('environment_variables', {}),
                enabled=config_data.get('enabled', True),
                tags=config_data.get('tags', []),
                metadata=config_data.get('metadata', {})
            )
            
            return workflow_config
            
        except Exception as e:
            logger.error(f"加载Workflow配置失败 {workflow_name}: {e}")
            raise
    
    def load_all_configs(self) -> None:
        """加载所有配置"""
        if not self.config_dir.exists():
            logger.warning(f"配置目录不存在: {self.config_dir}")
            return
        
        # 加载Agent配置
        agent_files = list(self.config_dir.glob("*agent*.yaml")) + list(self.config_dir.glob("*agent*.yml"))
        for config_file in agent_files:
            config_data = self.load_config_from_file(config_file.name)
            if config_data:
                agent_name = config_file.stem.replace('_agent', '').replace('-agent', '')
                try:
                    agent_config = self.load_agent_config(agent_name, config_data)
                    self.agents[agent_name] = agent_config
                    logger.info(f"加载Agent配置: {agent_name}")
                except Exception as e:
                    logger.error(f"Agent配置加载失败 {agent_name}: {e}")
        
        # 加载Workflow配置
        workflow_files = list(self.config_dir.glob("*workflow*.yaml")) + list(self.config_dir.glob("*workflow*.yml"))
        for config_file in workflow_files:
            config_data = self.load_config_from_file(config_file.name)
            if config_data:
                workflow_name = config_file.stem.replace('_workflow', '').replace('-workflow', '')
                try:
                    workflow_config = self.load_workflow_config(workflow_name, config_data)
                    self.workflows[workflow_name] = workflow_config
                    logger.info(f"加载Workflow配置: {workflow_name}")
                except Exception as e:
                    logger.error(f"Workflow配置加载失败 {workflow_name}: {e}")
        
        self._loaded = True
        logger.info(f"配置加载完成: {len(self.agents)} 个Agent, {len(self.workflows)} 个Workflow")
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """获取Agent配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.agents.get(agent_name)
    
    def get_workflow_config(self, workflow_name: str) -> Optional[WorkflowConfig]:
        """获取Workflow配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.workflows.get(workflow_name)
    
    def get_all_agents(self) -> Dict[str, AgentConfig]:
        """获取所有Agent配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.agents.copy()
    
    def get_all_workflows(self) -> Dict[str, WorkflowConfig]:
        """获取所有Workflow配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.workflows.copy()
    
    def list_agents(self) -> List[str]:
        """列出所有Agent名称"""
        if not self._loaded:
            self.load_all_configs()
        return list(self.agents.keys())
    
    def list_workflows(self) -> List[str]:
        """列出所有Workflow名称"""
        if not self._loaded:
            self.load_all_configs()
        return list(self.workflows.keys())
    
    def validate_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """验证Agent配置"""
        agent_config = self.get_agent_config(agent_name)
        if not agent_config:
            return {
                'is_valid': False,
                'errors': [f'Agent {agent_name} 不存在']
            }
        
        errors = []
        
        # 验证基本配置
        if not agent_config.name:
            errors.append('Agent名称不能为空')
        
        if not agent_config.task_categories:
            errors.append('必须至少指定一个任务类别')
        
        # 验证模型配置
        if not agent_config.model.name:
            errors.append('模型名称不能为空')
        
        # 验证工具配置
        enabled_tools = agent_config.get_enabled_tools()
        if not enabled_tools:
            errors.append('必须至少启用一个工具')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def reload_config(self, config_name: str, config_type: str = 'agent') -> bool:
        """重新加载指定配置"""
        try:
            if config_type == 'agent':
                config_file = f"{config_name}_agent.yaml"
                config_data = self.load_config_from_file(config_file)
                if config_data:
                    agent_config = self.load_agent_config(config_name, config_data)
                    self.agents[config_name] = agent_config
                    logger.info(f"重新加载Agent配置成功: {config_name}")
                    return True
            elif config_type == 'workflow':
                config_file = f"{config_name}_workflow.yaml"
                config_data = self.load_config_from_file(config_file)
                if config_data:
                    workflow_config = self.load_workflow_config(config_name, config_data)
                    self.workflows[config_name] = workflow_config
                    logger.info(f"重新加载Workflow配置成功: {config_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"重新加载配置失败 {config_name}: {e}")
            return False


# 全局配置管理器实例
_agent_config_manager = None


def get_agent_config_manager() -> AgentConfigManager:
    """获取全局Agent配置管理器实例"""
    global _agent_config_manager
    if _agent_config_manager is None:
        _agent_config_manager = AgentConfigManager()
    return _agent_config_manager 