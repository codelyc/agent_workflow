"""
基于SmolaAgents的微智能体

专业领域的具体执行任务，具有单一职责、工具专业化、可独立运行的特点
"""

import logging
import os
from typing import Dict, List, Any, Optional

from smolagents import CodeAgent, ToolCollection

from .base_micro_agent import BaseMicroAgent
from core.config.agent_config import AgentConfig, ToolConfig
from core.config.config_manager import ConfigManager
from core.config.llm_manager import get_llm_manager
from core.tracing import trace_event
from core.task_types.task_types import TaskType

logger = logging.getLogger(__name__)


class SmolaAgentsMicro(BaseMicroAgent):
    """基于SmolaAgents的微智能体"""
    
    def __init__(self, config: AgentConfig, config_manager: Optional[ConfigManager] = None):
        # 先设置配置，因为基类初始化会调用_get_tools()方法
        self.config = config
        self.config_manager = config_manager or ConfigManager()
        
        # 初始化基类
        super().__init__(
            model=config.model.name,
            max_steps=config.max_steps,
            timeout=config.timeout,
            workspace_path=config.workspace_path
        )
        
        # 创建SmolaAgents实例
        self._create_smolagents_instance()
        
        logger.info(f"初始化SmolaAgents微智能体: {self.config.name}")
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def description(self) -> str:
        return self.config.description
    
    @property
    def default_task_type(self) -> TaskType:
        return TaskType.from_string(self.config.default_task_type.value if self.config.default_task_type else "analysis")
    
    def _create_smolagents_instance(self):
        """创建SmolaAgents实例"""
        try:
            # 准备工具集合
            tools = self._prepare_tools()
            
            # 通过 LLMManager 创建模型对象
            llm_manager = get_llm_manager()
            model = llm_manager.create_smolagents_model(self.config.model.name)
            
            # 创建CodeAgent
            self.smolagent = CodeAgent(
                tools=tools,
                model=model,
                max_steps=self.config.max_steps,
                **self._get_smolagents_kwargs()
            )
            
            logger.info(f"SmolaAgents实例创建成功，使用模型: {self.config.model.name}")
            
        except Exception as e:
            logger.error(f"创建SmolaAgents实例失败: {e}")
            self.smolagent = None
    
    def _prepare_tools(self) -> List:
        """准备工具列表"""
        tools = []
        
        for tool_config in self.config.get_enabled_tools():
            try:
                tool_func = self._import_tool_function(tool_config)
                if tool_func:
                    tools.append(tool_func)
                    logger.debug(f"加载工具: {tool_config.name}")
            except Exception as e:
                logger.warning(f"加载工具失败 {tool_config.name}: {e}")
        
        return tools
    
    def _import_tool_function(self, tool_config: ToolConfig):
        """动态导入工具函数"""
        try:
            module_name = tool_config.module
            function_name = tool_config.function
            
            # 动态导入模块
            import importlib
            module = importlib.import_module(module_name)
            
            # 获取函数
            tool_func = getattr(module, function_name)
            
            return tool_func
            
        except Exception as e:
            logger.error(f"导入工具函数失败 {tool_config.module}.{tool_config.function}: {e}")
            return None
    
    def _get_smolagents_kwargs(self) -> Dict[str, Any]:
        """获取SmolaAgents参数"""
        kwargs = {}
        
        # 注意：CodeAgent不支持system_prompt等参数
        # 系统提示应该通过其他方式设置，比如在任务描述中包含
        
        return kwargs
    
    def _format_system_prompt(self) -> str:
        """格式化系统提示"""
        template = self.config.system_prompt_template
        if not template:
            return self._get_default_system_prompt()
        
        # 替换模板变量
        context = {
            'agent_name': self.config.name,
            'agent_description': self.config.description,
            'task_categories': ', '.join([cat.value for cat in self.config.task_categories]),
            'tools_list': ', '.join([tool.name for tool in self.config.get_enabled_tools()]),
            'default_task_type': self.config.default_task_type.value if self.config.default_task_type else 'analysis'
        }
        
        try:
            return template.format(**context)
        except Exception as e:
            logger.warning(f"格式化系统提示失败: {e}")
            return self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return f"""You are {self.config.name}, a specialized micro-agent for {self.config.description}.

Your primary focus is on {self.default_task_type.description}.

Available tools: {', '.join([tool.name for tool in self.config.get_enabled_tools()])}

Guidelines:
1. Focus on your specialized area of expertise
2. Use the most appropriate tools for each task
3. Provide detailed and accurate results
4. Handle errors gracefully and provide informative feedback
5. Be concise but thorough in your responses

Always respond in Chinese (中文) unless specifically requested otherwise."""
    
    def _get_tools(self) -> List:
        """获取工具列表 - 实现基类抽象方法"""
        return self._prepare_tools()
    
    def _get_system_prompt(self) -> str:
        """获取系统提示 - 实现基类抽象方法"""
        return self._get_default_system_prompt()
    
    def _execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行核心任务逻辑"""
        try:
            # 检查smolagent是否可用
            if self.smolagent is None:
                return f"智能体未正确初始化，无法执行任务: {task}"
            
            with trace_event("smolagents_micro_execution", "task_execution", 
                           agent=self.name, task=task):
                # 构建增强的任务描述
                enhanced_task = self._enhance_task_description(task, context)
                
                # 使用SmolaAgents执行
                result = self.smolagent.run(enhanced_task)
                
                # 将 RunResult 转换为字符串返回
                return str(result)
                
        except Exception as e:
            logger.error(f"SmolaAgents微智能体执行失败: {e}")
            return f"执行失败: {e}"
    
    def _enhance_task_description(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """增强任务描述"""
        enhanced = f"""
任务：{task}

执行上下文：
- 智能体：{self.config.name} ({self.config.description})
- 专业领域：{self.default_task_type.description}
- 工作空间：{self.workspace_path or '当前目录'}

可用工具：{', '.join([tool.name for tool in self.config.get_enabled_tools()])}
"""
        
        if context:
            enhanced += f"\n上下文信息：{context}\n"
        
        enhanced += """
请专注于你的专业领域，使用最合适的工具完成任务，并提供结构化的结果。
"""
        
        return enhanced
    
    def can_handle(self, subtask: Dict[str, Any]) -> bool:
        """检查是否能处理特定子任务"""
        # 检查任务类型匹配
        task_type = subtask.get("type", "")
        
        # 检查是否匹配配置的任务类别
        for category in self.config.task_categories:
            if category.value in task_type.lower():
                return True
        
        # 检查是否匹配智能体名称
        if self.name.lower() in task_type.lower():
            return True
        
        # 检查描述关键词
        description = subtask.get("description", "").lower()
        for tool in self.config.get_enabled_tools():
            if tool.name.lower() in description:
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        base_status = super().get_status()
        
        # 添加配置信息
        base_status.update({
            'config_name': self.config.name,
            'config_type': self.config.type.value,
            'smolagents_model': self.config.model.name,
            'enabled_tools': [tool.name for tool in self.config.get_enabled_tools()],
            'task_categories': [cat.value for cat in self.config.task_categories]
        })
        
        return base_status
    
    def __str__(self) -> str:
        return f"{self.config.name} (SmolaAgents微智能体)"
    
    def __repr__(self) -> str:
        return f"SmolaAgentsMicro(name='{self.config.name}', model='{self.config.model.name}')"