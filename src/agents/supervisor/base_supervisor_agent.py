"""
基础监督智能体

提供高级任务规划、工作流协调、结果整合等核心功能
集成SOP、管理微智能体、提供记忆系统
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from datetime import datetime

from core.task_types.task_types import TaskType
from core.tracing import task_context, sub_task_context, trace_event, get_current_task_id
from core.memory import get_memory_manager, MemoryManager

logger = logging.getLogger(__name__)


class BaseSupervisorAgent(ABC):
    """基础监督智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_steps: int = 50, 
                 timeout: int = 300, workspace_path: Optional[str] = None):
        self.model = model
        self.max_steps = max_steps
        self.timeout = timeout
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        
        # 初始化记忆管理器
        self.memory_manager = get_memory_manager(str(self.workspace_path))
        
        # 微智能体注册表
        self.micro_agents: Dict[str, Any] = {}
        
        # SOP内容缓存
        self._sop_content: Optional[str] = None
        
        logger.info(f"初始化监督智能体: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """智能体名称"""
        pass
    
    @property
    @abstractmethod
    def sop_category(self) -> str:
        """SOP分类目录名"""
        pass
    
    @property
    @abstractmethod
    def default_task_type(self) -> str:
        """默认任务类型"""
        pass
    
    @property
    def description(self) -> str:
        """智能体描述"""
        return f"{self.name} - 监督智能体，负责{self.default_task_type}类型任务的协调和管理"
    
    def register_micro_agent(self, agent_name: str, agent_instance: Any):
        """注册微智能体"""
        self.micro_agents[agent_name] = agent_instance
        logger.info(f"注册微智能体: {agent_name}")
    
    def get_micro_agent(self, agent_name: str) -> Optional[Any]:
        """获取微智能体"""
        return self.micro_agents.get(agent_name)
    
    def list_micro_agents(self) -> List[str]:
        """列出所有微智能体"""
        return list(self.micro_agents.keys())
    
    def _get_agent_worker_name(self) -> str:
        """根据智能体名称获取对应的AgentWorker目录名"""
        # 智能体名称到AgentWorker目录的映射
        name_mapping = {
            "code_assistant": "CodeAssistant",
            "data_analysis": "DataAnalysis", 
            "content_generation": "ContentGeneration"
        }
        
        # 尝试从智能体名称匹配
        agent_name = self.name.lower()
        for key, value in name_mapping.items():
            if key in agent_name:
                return value
        
        # 如果没有匹配，使用默认的转换规则
        # 将下划线命名转换为驼峰命名
        parts = agent_name.split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def load_sop(self) -> str:
        """加载SOP内容"""
        if self._sop_content is not None:
            return self._sop_content
        
        # 查找SOP文件 - 优先查找AgentWorker目录下的SOP
        # 根据智能体名称确定AgentWorker路径
        agent_worker_name = self._get_agent_worker_name()
        agent_sop_path = self.workspace_path / "AgentWorker" / agent_worker_name / "sop" / "sop.md"
        legacy_sop_path = self.workspace_path / "ai_agents" / "sop_workflows" / self.sop_category / "sop.md"
        
        # 优先使用新的AgentWorker路径结构
        sop_path = agent_sop_path if agent_sop_path.exists() else legacy_sop_path
        
        if not sop_path.exists():
            logger.warning(f"SOP文件不存在: {sop_path}")
            return self._get_default_sop()
        
        try:
            with open(sop_path, 'r', encoding='utf-8') as f:
                self._sop_content = f.read()
            
            logger.info(f"加载SOP: {sop_path}")
            return self._sop_content
            
        except Exception as e:
            logger.error(f"加载SOP失败: {e}")
            return self._get_default_sop()
    
    def _get_default_sop(self) -> str:
        """获取默认SOP"""
        return f"""
# {self.name} 默认工作流程

## 核心原则
- 明确任务边界和职责分工
- 采用增量式处理策略
- 重视结果验证和质量控制

## 工作流程
1. **初始评估**：分析当前状态和需求
2. **任务分解**：将复杂任务拆分为子任务
3. **智能体委派**：分配给合适的微智能体
4. **结果整合**：汇总和加工处理结果
5. **质量验证**：验证结果质量和完整性
6. **记忆更新**：更新系统记忆和状态

## 最佳实践
- 使用搜索而非遍历，提高效率
- 智能采样，避免过度处理
- 增量更新，避免重复工作
"""
    
    def run_with_tracing(self, task: str, task_id: Optional[str] = None) -> str:
        """带追踪的运行方法"""
        with task_context(task_id) as tid:
            return self.run(task, tid)
    
    def run(self, task: str, task_id: Optional[str] = None) -> str:
        """执行任务"""
        current_task_id = task_id or get_current_task_id()
        
        try:
            with trace_event("supervisor_task", "task_execution", task=task):
                logger.info(f"开始执行任务: {task[:100]}...")
                
                # 1. 现状评估
                current_state = self._assess_current_state(task)
                
                # 2. 检查是否需要执行
                if self._is_task_complete(current_state, task):
                    result = self._generate_summary(current_state)
                    self._update_memory(current_task_id, result, "completed_from_cache")
                    return result
                
                # 3. 任务分解与委派
                subtasks = self._decompose_task(task)
                results = []
                
                for i, subtask in enumerate(subtasks):
                    logger.info(f"执行子任务 {i+1}/{len(subtasks)}: {subtask['name']}")
                    
                    # 选择合适的微智能体
                    agent = self._select_micro_agent(subtask)
                    
                    if agent:
                        # 执行子任务
                        with sub_task_context(agent.name if hasattr(agent, 'name') else f"agent_{i}"):
                            result = self._execute_subtask(agent, subtask)
                            results.append({
                                'subtask': subtask,
                                'result': result,
                                'agent': agent.name if hasattr(agent, 'name') else str(agent)
                            })
                    else:
                        logger.warning(f"未找到合适的智能体执行子任务: {subtask['name']}")
                        results.append({
                            'subtask': subtask,
                            'result': f"无法执行子任务: {subtask['name']}",
                            'agent': 'none'
                        })
                
                # 4. 结果整合
                final_result = self._integrate_results(results, task)
                
                # 5. 更新记忆系统
                self._update_memory(current_task_id, final_result, "completed")
                
                logger.info("任务执行完成")
                return final_result
                
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            self._update_memory(current_task_id, str(e), "failed")
            raise
    
    def _assess_current_state(self, task: str) -> Dict[str, Any]:
        """评估当前状态"""
        with trace_event("assessment", "current_state_assessment"):
            # 获取项目记忆
            project_summary = self.memory_manager.get_project_summary()
            project_structure = self.memory_manager.get_project_structure()
            tech_stack = self.memory_manager.get_technology_stack()
            
            # 分析工作空间
            workspace_info = self._analyze_workspace()
            
            return {
                "project_summary": project_summary,
                "project_structure": project_structure,
                "technology_stack": tech_stack,
                "workspace_info": workspace_info,
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_workspace(self) -> Dict[str, Any]:
        """分析工作空间"""
        try:
            # 获取基本信息
            workspace_info = {
                "path": str(self.workspace_path),
                "exists": self.workspace_path.exists(),
                "is_dir": self.workspace_path.is_dir(),
                "files_count": 0,
                "directories_count": 0
            }
            
            if workspace_info["exists"] and workspace_info["is_dir"]:
                files = list(self.workspace_path.rglob("*"))
                workspace_info["files_count"] = len([f for f in files if f.is_file()])
                workspace_info["directories_count"] = len([f for f in files if f.is_dir()])
            
            return workspace_info
            
        except Exception as e:
            logger.warning(f"分析工作空间失败: {e}")
            return {"error": str(e)}
    
    def _is_task_complete(self, current_state: Dict[str, Any], task: str) -> bool:
        """检查任务是否已完成"""
        # 默认实现：检查是否有相关的任务结果缓存
        task_id = get_current_task_id()
        if task_id:
            cached_result = self.memory_manager.get_task_results(task_id)
            return cached_result is not None
        return False
    
    def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """分解任务为子任务"""
        # 默认实现：根据任务类型进行基本分解
        task_type = TaskType.from_string(self.default_task_type)
        
        if task_type == TaskType.ANALYSIS:
            return [
                {"name": "data_collection", "description": "收集相关数据和信息", "type": "search"},
                {"name": "analysis", "description": "分析收集到的数据", "type": "analysis"},
                {"name": "report_generation", "description": "生成分析报告", "type": "generation"}
            ]
        elif task_type == TaskType.CODE_GENERATION:
            return [
                {"name": "requirement_analysis", "description": "分析需求", "type": "analysis"},
                {"name": "code_generation", "description": "生成代码", "type": "generation"},
                {"name": "validation", "description": "验证代码质量", "type": "validation"}
            ]
        else:
            return [
                {"name": "general_task", "description": task, "type": "general"}
            ]
    
    def _select_micro_agent(self, subtask: Dict[str, Any]) -> Optional[Any]:
        """选择合适的微智能体"""
        # 默认实现：根据子任务类型选择智能体
        subtask_type = subtask.get("type", "general")
        
        # 优先选择已注册的微智能体
        for agent_name, agent in self.micro_agents.items():
            if hasattr(agent, 'can_handle') and agent.can_handle(subtask):
                return agent
            elif subtask_type in agent_name.lower():
                return agent
        
        # 如果没有找到特定的智能体，返回第一个可用的
        if self.micro_agents:
            return next(iter(self.micro_agents.values()))
        
        return None
    
    def _execute_subtask(self, agent: Any, subtask: Dict[str, Any]) -> str:
        """执行子任务"""
        try:
            if hasattr(agent, 'run'):
                return agent.run(subtask['description'])
            elif hasattr(agent, 'execute'):
                return agent.execute(subtask['description'])
            else:
                return f"智能体 {agent} 无法执行子任务"
                
        except Exception as e:
            logger.error(f"子任务执行失败: {e}")
            return f"子任务执行失败: {e}"
    
    @abstractmethod
    def _integrate_results(self, results: List[Dict[str, Any]], original_task: str) -> str:
        """整合结果 - 子类必须实现"""
        pass
    
    def _generate_summary(self, current_state: Dict[str, Any]) -> str:
        """生成摘要"""
        return f"""
基于当前状态的摘要报告：

项目概述: {current_state.get('project_summary', '未知')}
技术栈: {current_state.get('technology_stack', [])}
工作空间: {current_state.get('workspace_info', {}).get('path', '未知')}

分析时间: {current_state.get('timestamp', '未知')}
        """
    
    def _update_memory(self, task_id: Optional[str], result: str, status: str):
        """更新记忆系统"""
        if not task_id:
            return
        
        try:
            # 更新任务结果
            self.memory_manager.set_task_results(task_id, result)
            
            # 更新任务进度
            progress = {
                "status": status,
                "completed_at": datetime.now().isoformat(),
                "agent": self.name
            }
            self.memory_manager.set_task_progress(task_id, progress)
            
            logger.debug(f"更新任务记忆: {task_id}")
            
        except Exception as e:
            logger.warning(f"更新记忆失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "workspace_path": str(self.workspace_path),
            "micro_agents_count": len(self.micro_agents),
            "micro_agents": list(self.micro_agents.keys()),
            "sop_category": self.sop_category,
            "default_task_type": self.default_task_type
        }
    
    def __str__(self) -> str:
        return f"{self.name} (监督智能体)"
    
    def __repr__(self) -> str:
        return f"BaseSupervisorAgent(name='{self.name}', model='{self.model}')"