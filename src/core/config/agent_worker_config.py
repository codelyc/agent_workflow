"""
AgentWorker配置管理器

专门用于加载和管理AgentWorker目录下的YAML配置文件
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OutputControlConfig:
    """输出控制配置"""
    pre_constraints: List[Dict[str, Any]] = field(default_factory=list)
    self_check: Dict[str, Any] = field(default_factory=dict)
    post_processing: Dict[str, Any] = field(default_factory=dict)
    re_prompting: Dict[str, Any] = field(default_factory=dict)
    llm_client: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LLMConfig:
    """LLM配置"""
    default_model: str = "gpt-4o"
    backup_model: str = "gpt-4o-mini"
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    providers: Dict[str, Any] = field(default_factory=dict)
    openai: Dict[str, Any] = field(default_factory=dict)
    azure_openai: Dict[str, Any] = field(default_factory=dict)
    anthropic: Dict[str, Any] = field(default_factory=dict)
    caching: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolsConfig:
    """工具配置"""
    tools_config: Dict[str, Any] = field(default_factory=dict)
    data_sources: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    priorities: Dict[str, List[str]] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """代理配置"""
    agent_config: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentWorkerConfig:
    """AgentWorker完整配置"""
    worker_name: str
    output_control: Optional[OutputControlConfig] = None
    llm_config: Optional[LLMConfig] = None
    tools_config: Optional[ToolsConfig] = None
    agent_config: Optional[AgentConfig] = None
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径
        
        例如: get_config_value("llm_config.default_model")
        """
        parts = path.split('.')
        current = self.__dict__
        
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
            
            if current is None:
                return default
        
        return current


class AgentWorkerConfigManager:
    """AgentWorker配置管理器"""
    
    def __init__(self, agent_worker_dir: Union[str, Path] = "AgentWorker"):
        self.agent_worker_dir = Path(agent_worker_dir)
        self.configs: Dict[str, AgentWorkerConfig] = {}
        self._loaded = False
    
    def load_all_configs(self) -> None:
        """加载所有AgentWorker配置"""
        if not self.agent_worker_dir.exists():
            logger.warning(f"AgentWorker目录不存在: {self.agent_worker_dir}")
            return
        
        # 遍历所有Worker目录
        for worker_dir in self.agent_worker_dir.iterdir():
            if worker_dir.is_dir() and not worker_dir.name.startswith('.'):
                self._load_worker_config(worker_dir)
        
        self._loaded = True
        logger.info(f"AgentWorker配置加载完成: {len(self.configs)} 个Worker")
    
    def _load_worker_config(self, worker_dir: Path) -> None:
        """加载单个Worker的配置"""
        worker_name = worker_dir.name
        configs_dir = worker_dir / "configs"
        
        if not configs_dir.exists():
            logger.warning(f"Worker {worker_name} 缺少configs目录")
            return
        
        worker_config = AgentWorkerConfig(worker_name=worker_name)
        
        # 加载各种配置文件
        config_files = {
            'output_control.yaml': self._load_output_control_config,
            'llm_config.yaml': self._load_llm_config,
            'tools.yaml': self._load_tools_config,
            'analysis_agent.yaml': self._load_agent_config,
            'default_supervisor.yaml': self._load_agent_config,
        }
        
        for config_file, loader_func in config_files.items():
            config_path = configs_dir / config_file
            if config_path.exists():
                try:
                    config_data = loader_func(config_path)
                    if config_data:
                        setattr(worker_config, config_file.replace('.yaml', '').replace('-', '_'), config_data)
                except Exception as e:
                    logger.error(f"加载配置文件失败 {config_path}: {e}")
        
        self.configs[worker_name] = worker_config
        logger.info(f"加载Worker配置: {worker_name}")
    
    def _load_yaml_file(self, config_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"读取YAML文件失败 {config_path}: {e}")
            return {}
    
    def _load_output_control_config(self, config_path: Path) -> OutputControlConfig:
        """加载输出控制配置"""
        data = self._load_yaml_file(config_path)
        return OutputControlConfig(
            pre_constraints=data.get('pre_constraints', []),
            self_check=data.get('self_check', {}),
            post_processing=data.get('post_processing', {}),
            re_prompting=data.get('re_prompting', {}),
            llm_client=data.get('llm_client', {}),
            enabled=data.get('enabled', True)
        )
    
    def _load_llm_config(self, config_path: Path) -> LLMConfig:
        """加载LLM配置"""
        data = self._load_yaml_file(config_path)
        llm_config = data.get('llm_config', {})
        
        return LLMConfig(
            default_model=llm_config.get('default_model', 'gpt-4o'),
            backup_model=llm_config.get('backup_model', 'gpt-4o-mini'),
            model_parameters=llm_config.get('model_parameters', {}),
            providers=llm_config.get('providers', {}),
            openai=llm_config.get('openai', {}),
            azure_openai=llm_config.get('azure_openai', {}),
            anthropic=llm_config.get('anthropic', {}),
            caching=llm_config.get('caching', {}),
            rate_limits=llm_config.get('rate_limits', {})
        )
    
    def _load_tools_config(self, config_path: Path) -> ToolsConfig:
        """加载工具配置"""
        data = self._load_yaml_file(config_path)
        
        return ToolsConfig(
            tools_config=data.get('tools_config', {}),
            data_sources=data.get('data_sources', {}),
            performance=data.get('performance', {}),
            priorities=data.get('priorities', {}),
            security=data.get('security', {}),
            logging=data.get('logging', {})
        )
    
    def _load_agent_config(self, config_path: Path) -> AgentConfig:
        """加载代理配置"""
        data = self._load_yaml_file(config_path)
        
        return AgentConfig(
            agent_config=data.get('agent_config', {}),
            security=data.get('security', {}),
            monitoring=data.get('monitoring', {}),
            logging=data.get('logging', {})
        )
    
    def get_worker_config(self, worker_name: str) -> Optional[AgentWorkerConfig]:
        """获取指定Worker的配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.configs.get(worker_name)
    
    def get_all_configs(self) -> Dict[str, AgentWorkerConfig]:
        """获取所有Worker配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.configs.copy()
    
    def get_config_value(self, worker_name: str, config_path: str, default: Any = None) -> Any:
        """获取指定Worker的配置值
        
        Args:
            worker_name: Worker名称
            config_path: 配置路径，如 "llm_config.default_model"
            default: 默认值
        """
        worker_config = self.get_worker_config(worker_name)
        if not worker_config:
            return default
        
        return worker_config.get_config_value(config_path, default)
    
    def list_workers(self) -> List[str]:
        """列出所有可用的Worker"""
        if not self._loaded:
            self.load_all_configs()
        return list(self.configs.keys())
    
    def reload_config(self, worker_name: str) -> bool:
        """重新加载指定Worker的配置"""
        worker_dir = self.agent_worker_dir / worker_name
        if not worker_dir.exists():
            logger.error(f"Worker目录不存在: {worker_dir}")
            return False
        
        try:
            self._load_worker_config(worker_dir)
            logger.info(f"重新加载Worker配置成功: {worker_name}")
            return True
        except Exception as e:
            logger.error(f"重新加载Worker配置失败 {worker_name}: {e}")
            return False
    
    def validate_config(self, worker_name: str) -> Dict[str, Any]:
        """验证Worker配置
        
        Returns:
            验证结果，包含is_valid和errors字段
        """
        worker_config = self.get_worker_config(worker_name)
        if not worker_config:
            return {
                'is_valid': False,
                'errors': [f'Worker {worker_name} 不存在']
            }
        
        errors = []
        
        # 验证LLM配置
        if worker_config.llm_config:
            if not worker_config.llm_config.default_model:
                errors.append('LLM默认模型未配置')
        
        # 验证工具配置
        if worker_config.tools_config:
            tools_config = worker_config.tools_config.tools_config
            if 'enabled_categories' not in tools_config:
                errors.append('工具配置缺少enabled_categories')
        
        # 验证输出控制配置
        if worker_config.output_control:
            if not worker_config.output_control.enabled:
                logger.warning(f'Worker {worker_name} 的输出控制已禁用')
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> AgentWorkerConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = AgentWorkerConfigManager()
    return _config_manager


def load_worker_config(worker_name: str) -> Optional[AgentWorkerConfig]:
    """便捷函数：加载指定Worker配置"""
    return get_config_manager().get_worker_config(worker_name)


def get_worker_config_value(worker_name: str, config_path: str, default: Any = None) -> Any:
    """便捷函数：获取Worker配置值"""
    return get_config_manager().get_config_value(worker_name, config_path, default) 