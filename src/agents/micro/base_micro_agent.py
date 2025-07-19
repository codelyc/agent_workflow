"""
基础微智能体

专业领域的具体执行任务，具有单一职责、工具专业化、可独立运行的特点
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from core.task_types.task_types import TaskType
from core.tracing import trace_event, get_current_task_id, get_current_sub_task_id
from core.memory import get_memory_manager

logger = logging.getLogger(__name__)


class BaseMicroAgent(ABC):
    """基础微智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_steps: int = 20, 
                 timeout: int = 120, workspace_path: Optional[str] = None):
        self.model = model
        self.max_steps = max_steps
        self.timeout = timeout
        self.workspace_path = workspace_path
        
        # 获取记忆管理器
        self.memory_manager = get_memory_manager(workspace_path)
        
        # 工具注册表
        self._tools: List[Callable] = []
        self._initialize_tools()
        
        logger.info(f"初始化微智能体: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """智能体名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """智能体描述"""
        pass
    
    @property
    @abstractmethod
    def default_task_type(self) -> TaskType:
        """默认任务类型"""
        pass
    
    @property
    def capabilities(self) -> List[str]:
        """智能体能力列表"""
        return [tool.__name__ for tool in self._tools]
    
    def _initialize_tools(self):
        """初始化工具"""
        self._tools = self._get_tools()
        logger.debug(f"加载工具: {len(self._tools)} 个")
    
    @abstractmethod
    def _get_tools(self) -> List[Callable]:
        """获取工具列表 - 子类必须实现"""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """获取系统提示 - 子类必须实现"""
        pass
    
    def can_handle(self, subtask: Dict[str, Any]) -> bool:
        """检查是否能处理特定子任务"""
        # 默认实现：检查任务类型匹配
        task_type = subtask.get("type", "")
        
        # 检查是否匹配默认任务类型
        if self.default_task_type.value in task_type.lower():
            return True
        
        # 检查是否匹配智能体名称
        if self.name.lower() in task_type.lower():
            return True
        
        # 检查描述关键词
        description = subtask.get("description", "").lower()
        for capability in self.capabilities:
            if capability.lower() in description:
                return True
        
        return False
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行任务"""
        try:
            with trace_event("micro_agent_task", "task_execution", 
                           agent=self.name, task=task):
                logger.info(f"微智能体 {self.name} 开始执行: {task[:100]}...")
                
                # 1. 预处理
                processed_task = self._preprocess_task(task, context)
                
                # 2. 执行核心逻辑
                result = self._execute_task(processed_task, context)
                
                # 3. 后处理
                final_result = self._postprocess_result(result, context)
                
                # 4. 记录结果
                self._record_result(task, final_result)
                
                logger.info(f"微智能体 {self.name} 执行完成")
                return final_result
                
        except Exception as e:
            logger.error(f"微智能体 {self.name} 执行失败: {e}")
            self._record_error(task, str(e))
            raise
    
    def _preprocess_task(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """预处理任务"""
        # 默认实现：添加上下文信息
        if context:
            context_info = f"上下文信息: {context}\n\n"
            return context_info + task
        return task
    
    @abstractmethod
    def _execute_task(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """执行核心任务逻辑 - 子类必须实现"""
        pass
    
    def _postprocess_result(self, result: str, context: Optional[Dict[str, Any]]) -> str:
        """后处理结果"""
        # 默认实现：添加元信息
        metadata = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "task_id": get_current_task_id(),
            "sub_task_id": get_current_sub_task_id()
        }
        
        return f"{result}\n\n---\n执行信息: {metadata}"
    
    def _record_result(self, task: str, result: str):
        """记录执行结果"""
        try:
            sub_task_id = get_current_sub_task_id()
            if sub_task_id:
                # 记录子任务结果
                self.memory_manager.set_context_memory(
                    f"micro_agent_{self.name}_{sub_task_id}_result",
                    {
                        "task": task,
                        "result": result,
                        "agent": self.name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
        except Exception as e:
            logger.warning(f"记录结果失败: {e}")
    
    def _record_error(self, task: str, error: str):
        """记录执行错误"""
        try:
            sub_task_id = get_current_sub_task_id()
            if sub_task_id:
                # 记录错误信息
                self.memory_manager.set_context_memory(
                    f"micro_agent_{self.name}_{sub_task_id}_error",
                    {
                        "task": task,
                        "error": error,
                        "agent": self.name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
        except Exception as e:
            logger.warning(f"记录错误失败: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """获取特定工具"""
        for tool in self._tools:
            if tool.__name__ == tool_name:
                return tool
        return None
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        try:
            with trace_event("tool_execution", tool_name, agent=self.name):
                return tool(*args, **kwargs)
        except Exception as e:
            logger.error(f"工具执行失败 {tool_name}: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "default_task_type": self.default_task_type.value,
            "capabilities": self.capabilities,
            "tools_count": len(self._tools),
            "workspace_path": self.workspace_path
        }
    
    def get_help(self) -> str:
        """获取帮助信息"""
        help_text = f"""
{self.name} - 微智能体

描述: {self.description}
任务类型: {self.default_task_type.description}

可用工具:
"""
        for tool in self._tools:
            help_text += f"- {tool.__name__}: {tool.__doc__ or '无描述'}\n"
        
        return help_text
    
    def __str__(self) -> str:
        return f"{self.name} (微智能体)"
    
    def __repr__(self) -> str:
        return f"BaseMicroAgent(name='{self.name}', model='{self.model}')"


# 基础工具函数示例
def read_file_content(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {e}"


def write_file_content(file_path: str, content: str) -> str:
    """写入文件内容"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件: {file_path}"
    except Exception as e:
        return f"写入文件失败: {e}"


def list_directory(directory_path: str) -> str:
    """列出目录内容"""
    try:
        items = os.listdir(directory_path)
        return f"目录 {directory_path} 包含: {', '.join(items)}"
    except Exception as e:
        return f"列出目录失败: {e}"


def search_in_file(file_path: str, pattern: str) -> str:
    """在文件中搜索模式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        return f"在 {file_path} 中找到 {len(matches)} 个匹配: {matches[:10]}"  # 最多显示10个
    except Exception as e:
        return f"搜索失败: {e}"