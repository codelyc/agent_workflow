"""
工作流执行器

负责执行单个和批量工作流任务，提供完整的任务执行管理
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from ...core.lib.tracing import task_context, trace_event
from ...core.lib.memory import MemoryManager
from ...core.config.config_manager import ConfigManager
from ...core.config.agent_config import WorkflowConfig
from ...agents.factory.agent_factory import AgentFactory

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    task_description: str
    success: bool
    result: str
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算执行时间"""
        if self.start_time and self.end_time:
            self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class BatchResult:
    """批量任务执行结果"""
    batch_id: str
    total_tasks: int
    completed_tasks: int
    successful_tasks: int
    failed_tasks: int
    task_results: List[TaskResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    
    def __post_init__(self):
        """计算统计信息"""
        if self.start_time and self.end_time:
            self.total_execution_time = (self.end_time - self.start_time).total_seconds()
        
        self.completed_tasks = len(self.task_results)
        self.successful_tasks = len([r for r in self.task_results if r.success])
        self.failed_tasks = len([r for r in self.task_results if not r.success])
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.completed_tasks == 0:
            return 0.0
        return self.successful_tasks / self.completed_tasks
    
    def get_summary(self) -> str:
        """获取执行摘要"""
        summary = f"批量任务执行摘要 (ID: {self.batch_id})\n"
        summary += "=" * 50 + "\n"
        summary += f"总任务数: {self.total_tasks}\n"
        summary += f"完成任务数: {self.completed_tasks}\n"
        summary += f"成功任务数: {self.successful_tasks}\n"
        summary += f"失败任务数: {self.failed_tasks}\n"
        summary += f"成功率: {self.success_rate:.2%}\n"
        summary += f"总执行时间: {self.total_execution_time:.2f} 秒\n"
        
        if self.failed_tasks > 0:
            summary += "\n失败任务详情:\n"
            for result in self.task_results:
                if not result.success:
                    summary += f"- {result.task_id}: {result.error}\n"
        
        return summary


class WorkflowExecutor:
    """通用工作流执行器"""
    
    def __init__(self, 
                 agent_factory: Optional[AgentFactory] = None,
                 config_manager: Optional[ConfigManager] = None,
                 memory_manager: Optional[MemoryManager] = None):
        self.agent_factory = agent_factory or AgentFactory()
        self.config_manager = config_manager or ConfigManager()
        self.memory_manager = memory_manager or MemoryManager()
        
        logger.info("工作流执行器初始化完成")
    
    def execute_single_task(self, 
                           task: str, 
                           agent_name: str = "default_supervisor",
                           task_id: Optional[str] = None,
                           **kwargs) -> TaskResult:
        """执行单个任务"""
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"开始执行单任务: {task_id}")
        
        try:
            with task_context(task_id) as trace_id:
                with trace_event("workflow_execution", "single_task", 
                               task_id=task_id, agent=agent_name):
                    
                    # 获取智能体
                    agent = self.agent_factory.get_or_create_agent(agent_name)
                    if not agent:
                        raise ValueError(f"无法创建智能体: {agent_name}")
                    
                    # 执行任务
                    result = agent.run(task)
                    
                    end_time = datetime.now()
                    
                    # 创建任务结果
                    task_result = TaskResult(
                        task_id=task_id,
                        task_description=task,
                        success=True,
                        result=result,
                        start_time=start_time,
                        end_time=end_time,
                        agent_name=agent_name,
                        metadata=kwargs
                    )
                    
                    # 保存到记忆
                    self._save_task_result_to_memory(task_result)
                    
                    logger.info(f"任务执行成功: {task_id}")
                    return task_result
                    
        except Exception as e:
            end_time = datetime.now()
            error_msg = str(e)
            
            logger.error(f"任务执行失败: {task_id} - {error_msg}")
            
            task_result = TaskResult(
                task_id=task_id,
                task_description=task,
                success=False,
                result="",
                error=error_msg,
                start_time=start_time,
                end_time=end_time,
                agent_name=agent_name,
                metadata=kwargs
            )
            
            # 保存失败结果到记忆
            self._save_task_result_to_memory(task_result)
            
            return task_result
    
    def execute_batch_tasks(self, 
                           tasks: List[str], 
                           agent_name: str = "default_supervisor",
                           max_workers: int = 3,
                           batch_id: Optional[str] = None,
                           **kwargs) -> BatchResult:
        """批量执行任务"""
        batch_id = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"开始执行批量任务: {batch_id}, 任务数: {len(tasks)}")
        
        task_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(tasks):
                task_id = f"{batch_id}_task_{i+1}"
                future = executor.submit(
                    self.execute_single_task, 
                    task, 
                    agent_name, 
                    task_id, 
                    **kwargs
                )
                future_to_task[future] = (i, task, task_id)
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    task_result = future.result()
                    task_results.append(task_result)
                    
                    task_index, task_desc, task_id = future_to_task[future]
                    status = "成功" if task_result.success else "失败"
                    logger.info(f"任务 {task_index+1}/{len(tasks)} {status}: {task_id}")
                    
                except Exception as e:
                    task_index, task_desc, task_id = future_to_task[future]
                    logger.error(f"任务执行异常: {task_id} - {e}")
                    
                    # 创建失败结果
                    task_result = TaskResult(
                        task_id=task_id,
                        task_description=task_desc,
                        success=False,
                        result="",
                        error=str(e),
                        start_time=start_time,
                        end_time=datetime.now(),
                        agent_name=agent_name
                    )
                    task_results.append(task_result)
        
        end_time = datetime.now()
        
        # 创建批量结果
        batch_result = BatchResult(
            batch_id=batch_id,
            total_tasks=len(tasks),
            completed_tasks=0,  # 会在 __post_init__ 中计算
            successful_tasks=0,  # 会在 __post_init__ 中计算
            failed_tasks=0,     # 会在 __post_init__ 中计算
            task_results=task_results,
            start_time=start_time,
            end_time=end_time
        )
        
        # 保存批量结果到记忆
        self._save_batch_result_to_memory(batch_result)
        
        logger.info(f"批量任务执行完成: {batch_id}, 成功率: {batch_result.success_rate:.2%}")
        
        return batch_result
    
    def execute_workflow_config(self, 
                               workflow_name: str, 
                               input_data: Dict[str, Any],
                               **kwargs) -> Union[TaskResult, BatchResult]:
        """根据工作流配置执行"""
        workflow_config = self.config_manager.get_workflow_config(workflow_name)
        if not workflow_config:
            raise ValueError(f"工作流配置未找到: {workflow_name}")
        
        if not workflow_config.enabled:
            raise ValueError(f"工作流已禁用: {workflow_name}")
        
        logger.info(f"执行工作流配置: {workflow_name}")
        
        # 根据配置决定执行策略
        if workflow_config.execution_strategy == "sequential":
            return self._execute_sequential_workflow(workflow_config, input_data, **kwargs)
        elif workflow_config.execution_strategy == "parallel":
            return self._execute_parallel_workflow(workflow_config, input_data, **kwargs)
        elif workflow_config.execution_strategy == "conditional":
            return self._execute_conditional_workflow(workflow_config, input_data, **kwargs)
        else:
            raise ValueError(f"不支持的执行策略: {workflow_config.execution_strategy}")
    
    def _execute_sequential_workflow(self, 
                                   config: WorkflowConfig, 
                                   input_data: Dict[str, Any],
                                   **kwargs) -> TaskResult:
        """执行顺序工作流"""
        # 构建任务描述
        task_description = self._build_task_from_input(input_data, config)
        
        return self.execute_single_task(
            task=task_description,
            agent_name=config.supervisor_agent,
            **kwargs
        )
    
    def _execute_parallel_workflow(self, 
                                 config: WorkflowConfig, 
                                 input_data: Dict[str, Any],
                                 **kwargs) -> BatchResult:
        """执行并行工作流"""
        # 将输入数据分解为多个任务
        tasks = self._decompose_input_to_tasks(input_data, config)
        
        return self.execute_batch_tasks(
            tasks=tasks,
            agent_name=config.supervisor_agent,
            max_workers=config.max_concurrent_tasks,
            **kwargs
        )
    
    def _execute_conditional_workflow(self, 
                                    config: WorkflowConfig, 
                                    input_data: Dict[str, Any],
                                    **kwargs) -> Union[TaskResult, BatchResult]:
        """执行条件工作流"""
        # 简化实现：根据输入数据大小决定策略
        if len(str(input_data)) > 1000:  # 大任务，使用并行
            return self._execute_parallel_workflow(config, input_data, **kwargs)
        else:  # 小任务，使用顺序
            return self._execute_sequential_workflow(config, input_data, **kwargs)
    
    def _build_task_from_input(self, input_data: Dict[str, Any], config: WorkflowConfig) -> str:
        """从输入数据构建任务描述"""
        task_parts = []
        
        if "task" in input_data:
            task_parts.append(input_data["task"])
        
        if "requirements" in input_data:
            task_parts.append(f"要求: {input_data['requirements']}")
        
        if "context" in input_data:
            task_parts.append(f"上下文: {input_data['context']}")
        
        return "\n".join(task_parts) if task_parts else "默认任务：请处理提供的输入数据"
    
    def _decompose_input_to_tasks(self, input_data: Dict[str, Any], config: WorkflowConfig) -> List[str]:
        """将输入数据分解为多个任务"""
        tasks = []
        
        if "tasks" in input_data and isinstance(input_data["tasks"], list):
            return input_data["tasks"]
        
        # 默认分解策略
        base_task = self._build_task_from_input(input_data, config)
        
        # 如果有文件列表，为每个文件创建任务
        if "files" in input_data and isinstance(input_data["files"], list):
            for file_path in input_data["files"]:
                tasks.append(f"{base_task}\n处理文件: {file_path}")
        else:
            tasks.append(base_task)
        
        return tasks
    
    def _save_task_result_to_memory(self, task_result: TaskResult):
        """保存任务结果到记忆"""
        try:
            memory_data = {
                "task_id": task_result.task_id,
                "success": task_result.success,
                "execution_time": task_result.execution_time_seconds,
                "agent_name": task_result.agent_name,
                "timestamp": task_result.end_time.isoformat() if task_result.end_time else None
            }
            
            self.memory_manager.set_task_memory(
                task_result.task_id, 
                "result", 
                memory_data
            )
        except Exception as e:
            logger.warning(f"保存任务结果到记忆失败: {e}")
    
    def _save_batch_result_to_memory(self, batch_result: BatchResult):
        """保存批量结果到记忆"""
        try:
            memory_data = {
                "batch_id": batch_result.batch_id,
                "total_tasks": batch_result.total_tasks,
                "success_rate": batch_result.success_rate,
                "total_execution_time": batch_result.total_execution_time,
                "timestamp": batch_result.end_time.isoformat() if batch_result.end_time else None
            }
            
            self.memory_manager.set_task_memory(
                batch_result.batch_id, 
                "batch_result", 
                memory_data
            )
        except Exception as e:
            logger.warning(f"保存批量结果到记忆失败: {e}")
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取任务历史"""
        try:
            # 从任务存储中获取所有任务结果
            task_results = []
            for key, entry in self.memory_manager.task_store.memories.items():
                if key.endswith(".result"):
                    task_results.append({"key": key, **entry.value})
            
            # 按时间排序并限制数量
            sorted_tasks = sorted(
                task_results, 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )[:limit]
            
            return sorted_tasks
            
        except Exception as e:
            logger.error(f"获取任务历史失败: {e}")
            return []
    
    def get_batch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取批量任务历史"""
        try:
            # 从任务存储中获取所有批量结果
            batch_results = []
            for key, entry in self.memory_manager.task_store.memories.items():
                if key.endswith(".batch_result"):
                    batch_results.append({"key": key, **entry.value})
            
            # 按时间排序并限制数量
            sorted_batches = sorted(
                batch_results, 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )[:limit]
            
            return sorted_batches
            
        except Exception as e:
            logger.error(f"获取批量历史失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        try:
            task_history = self.get_task_history(100)  # 获取最近100个任务
            batch_history = self.get_batch_history(50)  # 获取最近50个批次
            
            total_tasks = len(task_history)
            successful_tasks = len([t for t in task_history if t.get("success", False)])
            
            total_batches = len(batch_history)
            avg_batch_success_rate = sum([b.get("success_rate", 0) for b in batch_history]) / max(total_batches, 1)
            
            return {
                "total_tasks_executed": total_tasks,
                "task_success_rate": successful_tasks / max(total_tasks, 1),
                "total_batches_executed": total_batches,
                "average_batch_success_rate": avg_batch_success_rate,
                "last_execution_time": task_history[0].get("timestamp") if task_history else None
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {} 