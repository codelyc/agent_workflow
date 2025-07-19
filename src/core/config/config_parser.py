"""
YAML 配置解析器

支持不同类型配置的专门解析，包括智能体配置、工作流配置、LLM配置等
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """配置类型枚举"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    LLM = "llm"
    SOP = "sop"
    SYSTEM = "system"


class AgentType(Enum):
    """智能体类型枚举"""
    SUPERVISOR = "supervisor"
    MICRO = "micro"


class WorkflowType(Enum):
    """工作流类型枚举"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    PROCESSING = "processing"
    CUSTOM = "custom"


@dataclass
class ConfigMetadata:
    """配置元数据"""
    config_type: ConfigType
    name: str
    version: str
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """智能体配置"""
    metadata: ConfigMetadata
    agent_type: AgentType
    enabled: bool = True
    default_task_type: str = "general"
    model: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    tracing: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 50
    max_retries: int = 3
    timeout: int = 300
    micro_agents: List[str] = field(default_factory=list)  # 仅监督智能体
    capabilities: List[str] = field(default_factory=list)  # 仅微智能体
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """工作流配置"""
    metadata: ConfigMetadata
    workflow_type: WorkflowType
    enabled: bool = True
    supervisor_agent: str = ""
    execution_strategy: str = "sequential"  # sequential, parallel, conditional
    max_concurrent_tasks: int = 3
    schedule_expression: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM配置"""
    metadata: ConfigMetadata
    default_provider: str = "openai"
    model_aliases: Dict[str, str] = field(default_factory=dict)
    providers: Dict[str, Any] = field(default_factory=dict)
    task_model_mapping: Dict[str, Any] = field(default_factory=dict)
    cost_control: Dict[str, Any] = field(default_factory=dict)
    load_balancing: Dict[str, Any] = field(default_factory=dict)
    caching: Dict[str, Any] = field(default_factory=dict)
    resilience: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SOPConfig:
    """SOP (标准作业程序) 配置"""
    metadata: ConfigMetadata
    sop_type: str = "custom"
    workflow_name: str = ""
    entry_point: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    agents: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_handling: Dict[str, Any] = field(default_factory=dict)


def load_tools_config(config_path: str = "configs/tools.yaml") -> List[str]:
    """
    Loads the tools configuration from a YAML file and returns a list of enabled tools.

    Args:
        config_path (str): The path to the tools YAML configuration file.

    Returns:
        List[str]: A list of tool names that are enabled in the configuration.
                   Returns an empty list if the file is not found or is invalid.
    """
    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Tools config file not found at: {config_path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict) or "tools" not in data:
            logger.error("Invalid tools config format. Root 'tools' key not found.")
            return []

        enabled_tools = [
            tool_name for tool_name, config in data["tools"].items() 
            if isinstance(config, dict) and config.get("enabled", False)
        ]
        
        return enabled_tools

    except yaml.YAMLError as e:
        logger.error(f"Error parsing tools YAML file: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading tools config: {e}")
        return []


class BaseConfigParser(ABC):
    """配置解析器基类"""
    
    def __init__(self, config_type: ConfigType):
        self.config_type = config_type
        self.schema = self._get_schema()
    
    @abstractmethod
    def _get_schema(self) -> Dict[str, Any]:
        """获取配置的JSON Schema"""
        pass
    
    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> Any:
        """解析配置数据"""
        pass
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证配置数据"""
        try:
            # 简化验证，只检查必需字段
            schema = self.schema
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    logger.error(f"缺少必需字段: {field}")
                    return False
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def _parse_metadata(self, data: Dict[str, Any]) -> ConfigMetadata:
        """解析配置元数据"""
        return ConfigMetadata(
            config_type=ConfigType(data.get("config_type", self.config_type.value)),
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            author=data.get("author"),
            tags=data.get("tags", [])
        )


class AgentConfigParser(BaseConfigParser):
    """智能体配置解析器"""
    
    def __init__(self):
        super().__init__(ConfigType.AGENT)
    
    def _get_schema(self) -> Dict[str, Any]:
        """智能体配置的JSON Schema"""
        return {
            "type": "object",
            "required": ["config_type", "agent_type", "name"],
            "properties": {
                "config_type": {"type": "string", "const": "agent"},
                "agent_type": {"type": "string", "enum": ["supervisor", "micro"]},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "enabled": {"type": "boolean"},
                "default_task_type": {"type": "string"},
                "model": {"type": "object"},
                "tools": {"type": "array", "items": {"type": "string"}},
                "memory": {"type": "object"},
                "tracing": {"type": "object"},
                "max_steps": {"type": "integer", "minimum": 1},
                "max_retries": {"type": "integer", "minimum": 0},
                "timeout": {"type": "integer", "minimum": 1},
                "micro_agents": {"type": "array", "items": {"type": "string"}},
                "capabilities": {"type": "array", "items": {"type": "string"}}
            }
        }
    
    def parse(self, data: Dict[str, Any]) -> AgentConfig:
        """解析智能体配置"""
        if not self.validate(data):
            raise ValueError("智能体配置验证失败")
        
        metadata = self._parse_metadata(data)
        agent_type = AgentType(data.get("agent_type", "micro"))
        
        return AgentConfig(
            metadata=metadata,
            agent_type=agent_type,
            enabled=data.get("enabled", True),
            default_task_type=data.get("default_task_type", "general"),
            model=data.get("model", {}),
            tools=data.get("tools", []),
            memory=data.get("memory", {}),
            tracing=data.get("tracing", {}),
            max_steps=data.get("max_steps", 50),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout", 300),
            micro_agents=data.get("micro_agents", []),
            capabilities=data.get("capabilities", []),
            custom_params={k: v for k, v in data.items() 
                          if k not in ["config_type", "agent_type", "name", "version", 
                                     "description", "enabled", "default_task_type", 
                                     "model", "tools", "memory", "tracing", "max_steps", 
                                     "max_retries", "timeout", "micro_agents", "capabilities"]}
        )


class WorkflowConfigParser(BaseConfigParser):
    """工作流配置解析器"""
    
    def __init__(self):
        super().__init__(ConfigType.WORKFLOW)
    
    def _get_schema(self) -> Dict[str, Any]:
        """工作流配置的JSON Schema"""
        return {
            "type": "object",
            "required": ["config_type", "workflow_type", "name"],
            "properties": {
                "config_type": {"type": "string", "const": "workflow"},
                "workflow_type": {"type": "string", "enum": ["analysis", "generation", "processing", "custom"]},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "enabled": {"type": "boolean"},
                "supervisor_agent": {"type": "string"},
                "execution_strategy": {"type": "string", "enum": ["sequential", "parallel", "conditional"]},
                "max_concurrent_tasks": {"type": "integer", "minimum": 1},
                "schedule_expression": {"type": ["string", "null"]},
                "parameters": {"type": "object"},
                "steps": {"type": "array"},
                "conditions": {"type": "object"},
                "error_handling": {"type": "object"},
                "outputs": {"type": "object"}
            }
        }
    
    def parse(self, data: Dict[str, Any]) -> WorkflowConfig:
        """解析工作流配置"""
        if not self.validate(data):
            raise ValueError("工作流配置验证失败")
        
        metadata = self._parse_metadata(data)
        workflow_type = WorkflowType(data.get("workflow_type", "custom"))
        
        return WorkflowConfig(
            metadata=metadata,
            workflow_type=workflow_type,
            enabled=data.get("enabled", True),
            supervisor_agent=data.get("supervisor_agent", ""),
            execution_strategy=data.get("execution_strategy", "sequential"),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 3),
            schedule_expression=data.get("schedule_expression"),
            parameters=data.get("parameters", {}),
            steps=data.get("steps", []),
            conditions=data.get("conditions", {}),
            error_handling=data.get("error_handling", {}),
            outputs=data.get("outputs", {})
        )


class LLMConfigParser(BaseConfigParser):
    """LLM配置解析器"""
    
    def __init__(self):
        super().__init__(ConfigType.LLM)
    
    def _get_schema(self) -> Dict[str, Any]:
        """LLM配置的JSON Schema"""
        return {
            "type": "object",
            "required": ["config_type"],
            "properties": {
                "config_type": {"type": "string", "const": "llm"},
                "version": {"type": "string"},
                "default_provider": {"type": "string"},
                "model_aliases": {"type": "object"},
                "openai": {"type": "object"},
                "anthropic": {"type": "object"},
                "azure_openai": {"type": "object"},
                "google": {"type": "object"},
                "zhipuai": {"type": "object"},
                "ollama": {"type": "object"},
                "task_model_mapping": {"type": "object"},
                "cost_control": {"type": "object"},
                "load_balancing": {"type": "object"},
                "caching": {"type": "object"},
                "resilience": {"type": "object"},
                "monitoring": {"type": "object"},
                "development": {"type": "object"}
            }
        }
    
    def parse(self, data: Dict[str, Any]) -> LLMConfig:
        """解析LLM配置"""
        if not self.validate(data):
            raise ValueError("LLM配置验证失败")
        
        metadata = self._parse_metadata(data)
        
        # 提取提供商配置
        providers = {}
        provider_keys = ["openai", "anthropic", "azure_openai", "google", "zhipuai", "ollama"]
        for key in provider_keys:
            if key in data:
                providers[key] = data[key]
        
        return LLMConfig(
            metadata=metadata,
            default_provider=data.get("default_provider", "openai"),
            model_aliases=data.get("model_aliases", {}),
            providers=providers,
            task_model_mapping=data.get("task_model_mapping", {}),
            cost_control=data.get("cost_control", {}),
            load_balancing=data.get("load_balancing", {}),
            caching=data.get("caching", {}),
            resilience=data.get("resilience", {}),
            monitoring=data.get("monitoring", {}),
            development=data.get("development", {})
        )


class SOPConfigParser(BaseConfigParser):
    """SOP配置解析器"""
    
    def __init__(self):
        super().__init__(ConfigType.SOP)
    
    def _get_schema(self) -> Dict[str, Any]:
        """SOP配置的JSON Schema"""
        return {
            "type": "object",
            "required": ["config_type", "name", "workflow_name"],
            "properties": {
                "config_type": {"type": "string", "const": "sop"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "sop_type": {"type": "string"},
                "workflow_name": {"type": "string"},
                "entry_point": {"type": "string"},
                "parameters": {"type": "object"},
                "agents": {"type": "array", "items": {"type": "string"}},
                "steps": {"type": "array"},
                "success_criteria": {"type": "object"},
                "failure_handling": {"type": "object"}
            }
        }
    
    def parse(self, data: Dict[str, Any]) -> SOPConfig:
        """解析SOP配置"""
        if not self.validate(data):
            raise ValueError("SOP配置验证失败")
        
        metadata = self._parse_metadata(data)
        
        return SOPConfig(
            metadata=metadata,
            sop_type=data.get("sop_type", "custom"),
            workflow_name=data.get("workflow_name", ""),
            entry_point=data.get("entry_point", ""),
            parameters=data.get("parameters", {}),
            agents=data.get("agents", []),
            steps=data.get("steps", []),
            success_criteria=data.get("success_criteria", {}),
            failure_handling=data.get("failure_handling", {})
        )


class ConfigParserFactory:
    """配置解析器工厂"""
    
    _parsers = {
        ConfigType.AGENT: AgentConfigParser,
        ConfigType.WORKFLOW: WorkflowConfigParser,
        ConfigType.LLM: LLMConfigParser,
        ConfigType.SOP: SOPConfigParser
    }
    
    @classmethod
    def get_parser(cls, config_type: ConfigType) -> BaseConfigParser:
        """获取指定类型的配置解析器"""
        parser_class = cls._parsers.get(config_type)
        if not parser_class:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        return parser_class()
    
    @classmethod
    def register_parser(cls, config_type: ConfigType, parser_class: Type[BaseConfigParser]):
        """注册新的配置解析器"""
        cls._parsers[config_type] = parser_class


class ConfigLoader:
    """统一配置加载器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.loaded_configs: Dict[str, Any] = {}
    
    def load_config_file(self, file_path: Union[str, Path]) -> Any:
        """加载单个配置文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 检查配置类型
            config_type_str = data.get("config_type")
            if not config_type_str:
                raise ValueError(f"配置文件缺少 config_type 字段: {file_path}")
            
            config_type = ConfigType(config_type_str)
            parser = ConfigParserFactory.get_parser(config_type)
            
            config = parser.parse(data)
            
            # 缓存配置
            config_name = config.metadata.name
            self.loaded_configs[config_name] = config
            
            logger.info(f"配置加载成功: {config_name} ({config_type.value})")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {file_path} - {e}")
            raise
    
    def load_configs_from_directory(self, subdir: str = "") -> Dict[str, Any]:
        """从目录加载所有配置文件"""
        target_dir = self.config_dir / subdir if subdir else self.config_dir
        
        if not target_dir.exists():
            logger.warning(f"配置目录不存在: {target_dir}")
            return {}
        
        configs = {}
        
        for file_path in target_dir.glob("**/*.yaml"):
            try:
                config = self.load_config_file(file_path)
                configs[config.metadata.name] = config
            except Exception as e:
                logger.error(f"跳过配置文件 {file_path}: {e}")
        
        logger.info(f"从目录 {target_dir} 加载了 {len(configs)} 个配置")
        return configs
    
    def get_config(self, name: str) -> Optional[Any]:
        """获取已加载的配置"""
        return self.loaded_configs.get(name)
    
    def get_configs_by_type(self, config_type: ConfigType) -> List[Any]:
        """获取指定类型的所有配置"""
        return [config for config in self.loaded_configs.values()
                if config.metadata.config_type == config_type]
    
    def list_configs(self) -> Dict[str, str]:
        """列出所有已加载的配置"""
        return {name: config.metadata.config_type.value 
                for name, config in self.loaded_configs.items()}
    
    def validate_all_configs(self) -> List[str]:
        """验证所有配置"""
        issues = []
        
        for name, config in self.loaded_configs.items():
            try:
                # 重新验证配置
                config_type = config.metadata.config_type
                parser = ConfigParserFactory.get_parser(config_type)
                
                # 这里需要原始数据来验证，实际应用中可以存储原始数据
                logger.info(f"配置 {name} 验证通过")
                
            except Exception as e:
                issue = f"配置 {name} 验证失败: {e}"
                issues.append(issue)
                logger.error(issue)
        
        return issues
    
    def export_config(self, name: str, output_path: Union[str, Path]) -> bool:
        """导出配置到文件"""
        config = self.get_config(name)
        if not config:
            logger.error(f"配置不存在: {name}")
            return False
        
        try:
            # 将配置对象转换回字典格式
            config_dict = self._config_to_dict(config)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置导出成功: {name} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            return False
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        # 这是一个简化的实现，实际应用中可能需要更复杂的转换逻辑
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__') and hasattr(value, 'config_type'):
                    # 处理元数据对象
                    result.update(self._config_to_dict(value))
                else:
                    result[key] = value
            return result
        return config


# 全局配置加载器实例
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: str = "configs") -> ConfigLoader:
    """获取全局配置加载器"""
    global _global_config_loader
    
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_dir)
    
    return _global_config_loader


def reset_config_loader():
    """重置全局配置加载器"""
    global _global_config_loader
    _global_config_loader = None 