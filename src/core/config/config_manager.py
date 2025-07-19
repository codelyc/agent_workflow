"""
配置管理器

支持YAML配置文件的加载、验证和管理
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .agent_config import (
    AgentConfig, WorkflowConfig, ModelConfig, ToolConfig, 
    MemoryConfig, TracingConfig, AgentType, TaskCategory
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.agents_config: Dict[str, AgentConfig] = {}
        self.workflows_config: Dict[str, WorkflowConfig] = {}
        self._loaded = False
    
    def load_all_configs(self) -> None:
        """加载所有配置文件"""
        if not self.config_dir.exists():
            logger.warning(f"配置目录不存在: {self.config_dir}")
            return
        
        # 加载智能体配置
        agents_dir = self.config_dir / "agents"
        if agents_dir.exists():
            self._load_agents_configs(agents_dir)
        
        # 加载工作流配置
        workflows_dir = self.config_dir / "workflows"
        if workflows_dir.exists():
            self._load_workflows_configs(workflows_dir)
        
        self._loaded = True
        logger.info(f"配置加载完成: {len(self.agents_config)} 个智能体, {len(self.workflows_config)} 个工作流")
    
    def _load_agents_configs(self, agents_dir: Path) -> None:
        """加载智能体配置"""
        for config_file in agents_dir.glob("*.yaml"):
            try:
                agent_config = self.load_agent_config(config_file)
                if agent_config:
                    self.agents_config[agent_config.name] = agent_config
                    logger.debug(f"加载智能体配置: {agent_config.name}")
            except Exception as e:
                logger.error(f"加载智能体配置失败 {config_file}: {e}")
    
    def _load_workflows_configs(self, workflows_dir: Path) -> None:
        """加载工作流配置"""
        for config_file in workflows_dir.glob("*.yaml"):
            try:
                workflow_config = self.load_workflow_config(config_file)
                if workflow_config:
                    self.workflows_config[workflow_config.name] = workflow_config
                    logger.debug(f"加载工作流配置: {workflow_config.name}")
            except Exception as e:
                logger.error(f"加载工作流配置失败 {config_file}: {e}")
    
    def load_agent_config(self, config_path: Union[str, Path]) -> Optional[AgentConfig]:
        """加载单个智能体配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return self._parse_agent_config(data)
            
        except Exception as e:
            logger.error(f"加载智能体配置失败 {config_path}: {e}")
            return None
    
    def load_workflow_config(self, config_path: Union[str, Path]) -> Optional[WorkflowConfig]:
        """加载单个工作流配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return self._parse_workflow_config(data)
            
        except Exception as e:
            logger.error(f"加载工作流配置失败 {config_path}: {e}")
            return None
    
    def _parse_agent_config(self, data: Dict[str, Any]) -> AgentConfig:
        """解析智能体配置数据"""
        # 基础信息
        name = data.get('name', '')
        agent_type = AgentType(data.get('type', 'micro'))
        description = data.get('description', '')
        
        # 任务配置
        task_categories = []
        for cat_str in data.get('task_categories', []):
            try:
                task_categories.append(TaskCategory(cat_str))
            except ValueError:
                logger.warning(f"未知任务类别: {cat_str}")
        
        default_task_type = None
        if data.get('default_task_type'):
            try:
                default_task_type = TaskCategory(data['default_task_type'])
            except ValueError:
                logger.warning(f"未知默认任务类型: {data['default_task_type']}")
        
        # 模型配置
        model_data = data.get('model', {})
        model = ModelConfig(
            name=model_data.get('name', 'gpt-4o-mini'),
            api_key=model_data.get('api_key'),
            base_url=model_data.get('base_url'),
            max_tokens=model_data.get('max_tokens', 4000),
            temperature=model_data.get('temperature', 0.1),
            timeout=model_data.get('timeout', 30),
            retry_attempts=model_data.get('retry_attempts', 3)
        )
        
        # 工具配置
        tools = []
        for tool_data in data.get('tools', []):
            tool = ToolConfig(
                name=tool_data['name'],
                module=tool_data['module'],
                function=tool_data['function'],
                description=tool_data.get('description', ''),
                parameters=tool_data.get('parameters', {}),
                enabled=tool_data.get('enabled', True)
            )
            tools.append(tool)
        
        # 记忆配置
        memory_data = data.get('memory', {})
        memory = MemoryConfig(
            enabled=memory_data.get('enabled', True),
            storage_path=memory_data.get('storage_path'),
            cache_size=memory_data.get('cache_size', 1000),
            cleanup_interval=memory_data.get('cleanup_interval', 3600)
        )
        
        # 追踪配置
        tracing_data = data.get('tracing', {})
        tracing = TracingConfig(
            enabled=tracing_data.get('enabled', True),
            langfuse_public_key=tracing_data.get('langfuse_public_key'),
            langfuse_private_key=tracing_data.get('langfuse_private_key'),
            langfuse_host=tracing_data.get('langfuse_host'),
            log_level=tracing_data.get('log_level', 'INFO')
        )
        
        return AgentConfig(
            name=name,
            type=agent_type,
            description=description,
            task_categories=task_categories,
            default_task_type=default_task_type,
            model=model,
            tools=tools,
            max_steps=data.get('max_steps', 50),
            timeout=data.get('timeout', 300),
            max_retries=data.get('max_retries', 3),
            sop_category=data.get('sop_category'),
            sop_path=data.get('sop_path'),
            micro_agents=data.get('micro_agents', []),
            system_prompt=data.get('system_prompt'),
            system_prompt_template=data.get('system_prompt_template'),
            memory=memory,
            tracing=tracing,
            workspace_path=data.get('workspace_path'),
            enabled=data.get('enabled', True),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
    
    def _parse_workflow_config(self, data: Dict[str, Any]) -> WorkflowConfig:
        """解析工作流配置数据"""
        return WorkflowConfig(
            name=data.get('name', ''),
            description=data.get('description', ''),
            supervisor_agent=data.get('supervisor_agent', ''),
            micro_agents=data.get('micro_agents', []),
            execution_strategy=data.get('execution_strategy', 'sequential'),
            max_concurrent_tasks=data.get('max_concurrent_tasks', 3),
            timeout=data.get('timeout', 600),
            input_schema=data.get('input_schema', {}),
            output_format=data.get('output_format', 'markdown'),
            workspace_path=data.get('workspace_path'),
            environment_variables=data.get('environment_variables', {}),
            enabled=data.get('enabled', True),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
    
    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """获取智能体配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.agents_config.get(name)
    
    def get_workflow_config(self, name: str) -> Optional[WorkflowConfig]:
        """获取工作流配置"""
        if not self._loaded:
            self.load_all_configs()
        return self.workflows_config.get(name)
    
    def list_agents(self, agent_type: Optional[AgentType] = None, 
                   enabled_only: bool = True) -> List[str]:
        """列出智能体名称"""
        if not self._loaded:
            self.load_all_configs()
        
        result = []
        for name, config in self.agents_config.items():
            if enabled_only and not config.enabled:
                continue
            if agent_type and config.type != agent_type:
                continue
            result.append(name)
        
        return result
    
    def list_workflows(self, enabled_only: bool = True) -> List[str]:
        """列出工作流名称"""
        if not self._loaded:
            self.load_all_configs()
        
        result = []
        for name, config in self.workflows_config.items():
            if enabled_only and not config.enabled:
                continue
            result.append(name)
        
        return result
    
    def get_agents_by_task_category(self, task_category: TaskCategory, 
                                   agent_type: Optional[AgentType] = None) -> List[str]:
        """根据任务类别获取智能体"""
        if not self._loaded:
            self.load_all_configs()
        
        result = []
        for name, config in self.agents_config.items():
            if not config.enabled:
                continue
            if agent_type and config.type != agent_type:
                continue
            if config.can_handle_task(task_category):
                result.append(name)
        
        return result
    
    def save_agent_config(self, config: AgentConfig, 
                         config_path: Optional[Union[str, Path]] = None) -> bool:
        """保存智能体配置"""
        try:
            if config_path is None:
                agents_dir = self.config_dir / "agents"
                agents_dir.mkdir(parents=True, exist_ok=True)
                config_path = agents_dir / f"{config.name}.yaml"
            
            config_dict = config.to_dict()
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            # 更新内存中的配置
            self.agents_config[config.name] = config
            
            logger.info(f"保存智能体配置: {config.name} -> {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存智能体配置失败 {config.name}: {e}")
            return False
    
    def create_default_config_files(self) -> None:
        """创建默认配置文件"""
        # 创建配置目录
        self.config_dir.mkdir(parents=True, exist_ok=True)
        agents_dir = self.config_dir / "agents"
        workflows_dir = self.config_dir / "workflows"
        agents_dir.mkdir(exist_ok=True)
        workflows_dir.mkdir(exist_ok=True)
        
        # 创建默认监督智能体配置
        default_supervisor = AgentConfig(
            name="default_supervisor",
            type=AgentType.SUPERVISOR,
            description="默认监督智能体，负责任务协调和结果整合",
            task_categories=[TaskCategory.ANALYSIS, TaskCategory.GENERATION],
            default_task_type=TaskCategory.ANALYSIS,
            sop_category="custom_workflow",
            micro_agents=["search_agent", "analysis_agent"],
            tools=[
                ToolConfig(
                    name="read_file",
                    module="ai_agents.tools.file_tools",
                    function="read_file_content",
                    description="读取文件内容"
                ),
                ToolConfig(
                    name="list_directory",
                    module="ai_agents.tools.file_tools", 
                    function="list_directory",
                    description="列出目录内容"
                )
            ]
        )
        
        # 创建默认搜索微智能体配置
        search_agent = AgentConfig(
            name="search_agent",
            type=AgentType.MICRO,
            description="搜索智能体，专门处理文件搜索和信息检索任务",
            task_categories=[TaskCategory.SEARCH],
            default_task_type=TaskCategory.SEARCH,
            tools=[
                ToolConfig(
                    name="search_files",
                    module="ai_agents.tools.file_tools",
                    function="search_files",
                    description="搜索文件"
                ),
                ToolConfig(
                    name="find_code_patterns",
                    module="ai_agents.tools.code_tools",
                    function="find_code_patterns",
                    description="查找代码模式"
                )
            ]
        )
        
        # 创建默认分析微智能体配置
        analysis_agent = AgentConfig(
            name="analysis_agent",
            type=AgentType.MICRO,
            description="分析智能体，专门处理代码分析和质量评估任务",
            task_categories=[TaskCategory.ANALYSIS, TaskCategory.VALIDATION],
            default_task_type=TaskCategory.ANALYSIS,
            tools=[
                ToolConfig(
                    name="parse_python_code",
                    module="ai_agents.tools.code_tools",
                    function="parse_python_code",
                    description="解析Python代码结构"
                ),
                ToolConfig(
                    name="analyze_imports",
                    module="ai_agents.tools.code_tools",
                    function="analyze_imports",
                    description="分析导入依赖"
                )
            ]
        )
        
        # 保存配置
        self.save_agent_config(default_supervisor)
        self.save_agent_config(search_agent)
        self.save_agent_config(analysis_agent)
        
        logger.info("默认配置文件创建完成")


# 便捷函数
def load_agent_config(config_path: Union[str, Path]) -> Optional[AgentConfig]:
    """加载单个智能体配置"""
    manager = ConfigManager()
    return manager.load_agent_config(config_path)


def load_workflow_config(config_path: Union[str, Path]) -> Optional[WorkflowConfig]:
    """加载单个工作流配置"""
    manager = ConfigManager()
    return manager.load_workflow_config(config_path) 