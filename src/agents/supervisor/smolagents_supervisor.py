"""
基于SmolaAgents的监督智能体

集成YAML配置，提供SOP驱动的工作流管理和微智能体协调
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from smolagents import CodeAgent, ToolCollection

from .base_supervisor_agent import BaseSupervisorAgent
from core.config.agent_config import AgentConfig, ToolConfig, AgentType, TaskCategory
from core.config.config_manager import ConfigManager
from core.config.llm_manager import get_llm_manager
from core.tracing import task_context, sub_task_context, trace_event

logger = logging.getLogger(__name__)


class SmolaAgentsSupervisor(BaseSupervisorAgent):
    """基于SmolaAgents的监督智能体"""
    
    def __init__(self, config: AgentConfig, config_manager: Optional[ConfigManager] = None):
        # 先设置配置，因为基类初始化会调用 name 属性
        self.config = config
        self.config_manager = config_manager or ConfigManager()
        
        # 初始化基类
        super().__init__(
            model=config.model.name,
            max_steps=getattr(config, 'max_steps', 10),
            timeout=getattr(config, 'timeout', 300),
            workspace_path=getattr(config, 'workspace_path', None)
        )
        
        # 创建SmolaAgents实例
        self._create_smolagents_instance()
        
        # 加载和注册微智能体
        self._load_micro_agents()
        
        logger.info(f"初始化SmolaAgents监督智能体: {self.config.name}")
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def sop_category(self) -> str:
        return self.config.sop_category or "custom_workflow"
    
    @property
    def default_task_type(self) -> str:
        return self.config.default_task_type.value if self.config.default_task_type else "analysis"
    
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
            return f"You are {self.config.name}, {self.config.description}"
        
        # 替换模板变量
        context = {
            'agent_name': self.config.name,
            'agent_description': self.config.description,
            'task_categories': ', '.join([cat.value for cat in self.config.task_categories]),
            'sop_content': self.load_sop(),
            'tools_list': ', '.join([tool.name for tool in self.config.get_enabled_tools()])
        }
        
        try:
            return template.format(**context)
        except Exception as e:
            logger.warning(f"格式化系统提示失败: {e}")
            return template
    
    def _load_micro_agents(self):
        """加载和注册微智能体"""
        for agent_name in self.config.micro_agents:
            try:
                agent_config = self.config_manager.get_agent_config(agent_name)
                if agent_config and agent_config.enabled:
                    # 创建微智能体实例
                    from ..micro.smolagents_micro import SmolaAgentsMicro
                    micro_agent = SmolaAgentsMicro(agent_config, self.config_manager)
                    self.register_micro_agent(agent_name, micro_agent)
                    logger.debug(f"注册微智能体: {agent_name}")
                else:
                    logger.warning(f"微智能体配置未找到或已禁用: {agent_name}")
            except Exception as e:
                logger.error(f"加载微智能体失败 {agent_name}: {e}")
    
    def _integrate_results(self, results: List[Dict[str, Any]], original_task: str) -> str:
        """整合结果"""
        try:
            with trace_event("result_integration", "integrate_supervisor_results"):
                # 构建结果摘要
                integrated_result = f"# {self.config.name} 执行结果\n\n"
                integrated_result += f"**任务**: {original_task}\n\n"
                integrated_result += f"**执行时间**: {self._get_current_time()}\n\n"
                
                # 添加执行摘要
                integrated_result += "## 执行摘要\n\n"
                successful_tasks = len([r for r in results if not r['result'].startswith('失败')])
                integrated_result += f"- 总任务数: {len(results)}\n"
                integrated_result += f"- 成功完成: {successful_tasks}\n"
                integrated_result += f"- 失败任务: {len(results) - successful_tasks}\n\n"
                
                # 添加详细结果
                integrated_result += "## 详细结果\n\n"
                for i, result_data in enumerate(results, 1):
                    subtask = result_data['subtask']
                    result = result_data['result']
                    agent = result_data['agent']
                    
                    integrated_result += f"### {i}. {subtask['name']}\n\n"
                    integrated_result += f"**执行智能体**: {agent}\n\n"
                    integrated_result += f"**任务描述**: {subtask['description']}\n\n"
                    integrated_result += f"**执行结果**:\n\n{result}\n\n"
                    integrated_result += "---\n\n"
                
                # 添加SOP信息
                sop_content = self.load_sop()
                if sop_content:
                    integrated_result += "## 工作流程说明\n\n"
                    integrated_result += "本次执行遵循以下SOP流程:\n\n"
                    integrated_result += "```markdown\n"
                    integrated_result += sop_content[:500] + "...\n"  # 截取前500字符
                    integrated_result += "```\n\n"
                
                # 添加改进建议
                integrated_result += "## 总结与建议\n\n"
                if successful_tasks == len(results):
                    integrated_result += "✅ 所有任务执行成功，达到预期目标。\n\n"
                else:
                    integrated_result += "⚠️ 部分任务执行失败，建议:\n"
                    integrated_result += "- 检查失败任务的错误信息\n"
                    integrated_result += "- 确认环境配置和权限设置\n"
                    integrated_result += "- 考虑调整任务分解策略\n\n"
                
                integrated_result += f"*报告生成时间: {self._get_current_time()}*\n"
                integrated_result += f"*执行智能体: {self.config.name}*\n"
                
                return integrated_result
                
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            # 简单的结果连接作为备选
            return "\n\n".join([r['result'] for r in results])
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def run(self, task: str, task_id: Optional[str] = None) -> str:
        """执行任务 - 优先使用SmolaAgents，失败时降级到基础实现"""
        try:
            # 检查smolagent是否可用
            if self.smolagent is None:
                logger.warning("SmolaAgents实例不可用，使用基础实现")
                return super().run(task, task_id)
            
            with trace_event("smolagents_execution", "supervisor_run", task=task):
                # 构建增强的任务描述
                enhanced_task = self._enhance_task_description(task)
                
                # 使用SmolaAgents执行
                result = self.smolagent.run(enhanced_task)
                
                # 将RunResult转换为字符串
                return str(result)
                
        except Exception as e:
            logger.error(f"SmolaAgents执行失败: {e}")
            # 降级到基础实现
            return super().run(task, task_id)
    
    def run_with_smolagents(self, task: str) -> str:
        """使用SmolaAgents运行任务（向后兼容方法）"""
        return self.run(task)
    
    def _enhance_task_description(self, task: str) -> str:
        """增强任务描述"""
        enhanced = f"""
任务：{task}

执行上下文：
- 智能体：{self.config.name} ({self.config.description})
- 任务类型：{self.default_task_type}
- 工作空间：{self.workspace_path or '当前目录'}

可用工具：{', '.join([tool.name for tool in self.config.get_enabled_tools()])}

请根据SOP流程执行此任务，确保：
1. 分析当前状态和需求
2. 合理分解任务
3. 选择合适的工具
4. 提供结构化的结果

"""
        return enhanced
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        base_status = super().get_status()
        
        # 添加配置信息
        base_status.update({
            'config_name': self.config.name,
            'config_type': self.config.type.value,
            'smolagents_model': self.config.model.name,
            'enabled_tools': [tool.name for tool in self.config.get_enabled_tools()],
            'micro_agents_count': len(self.config.micro_agents),
            'task_categories': [cat.value for cat in self.config.task_categories]
        })
        
        return base_status
    
    def __str__(self) -> str:
        return f"{self.config.name} (SmolaAgents监督智能体)"
    
    def __repr__(self) -> str:
        return f"SmolaAgentsSupervisor(name='{self.config.name}', model='{self.config.model.name}')"