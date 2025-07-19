"""
自动化工作流管理器

负责注册、管理和执行自动化工作流，支持调度和批量处理
"""

import logging
import schedule
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Thread
import json

from ..engine.workflow_executor import WorkflowExecutor, TaskResult, BatchResult
from ...agents.factory.agent_factory import AgentFactory
from ...core.config.config_manager import ConfigManager
from ...core.config.agent_config import WorkflowConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkflowRegistration:
    """工作流注册信息"""
    name: str
    supervisor_class: type
    config: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[WorkflowExecutor] = None
    enabled: bool = True
    description: str = ""
    schedule_expression: Optional[str] = None  # cron 表达式
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    workflow_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomatedWorkflowManager:
    """自动化工作流管理器"""
    
    def __init__(self, 
                 agent_factory: Optional[AgentFactory] = None,
                 config_manager: Optional[ConfigManager] = None):
        self.agent_factory = agent_factory or AgentFactory()
        self.config_manager = config_manager or ConfigManager()
        
        # 工作流注册表
        self.workflows: Dict[str, WorkflowRegistration] = {}
        
        # 执行历史
        self.execution_history: List[WorkflowExecution] = []
        
        # 调度器
        self.scheduler_running = False
        self.scheduler_thread: Optional[Thread] = None
        
        logger.info("自动化工作流管理器初始化完成")
    
    def register_workflow(self, 
                         name: str, 
                         supervisor_class: type, 
                         config: Optional[Dict[str, Any]] = None,
                         description: str = "",
                         schedule_expression: Optional[str] = None) -> WorkflowRegistration:
        """注册工作流"""
        config = config or {}
        
        # 创建执行器
        executor = WorkflowExecutor(self.agent_factory, self.config_manager)
        
        # 创建注册信息
        registration = WorkflowRegistration(
            name=name,
            supervisor_class=supervisor_class,
            config=config,
            executor=executor,
            description=description,
            schedule_expression=schedule_expression
        )
        
        self.workflows[name] = registration
        
        # 如果有调度表达式，添加到调度器
        if schedule_expression:
            self._add_to_scheduler(registration)
        
        logger.info(f"注册工作流: {name}")
        return registration
    
    def register_workflow_from_config(self, workflow_name: str) -> Optional[WorkflowRegistration]:
        """从配置文件注册工作流"""
        workflow_config = self.config_manager.get_workflow_config(workflow_name)
        if not workflow_config:
            logger.error(f"工作流配置未找到: {workflow_name}")
            return None
        
        if not workflow_config.enabled:
            logger.warning(f"工作流已禁用: {workflow_name}")
            return None
        
        # 动态获取监督智能体类
        supervisor_agent = self.agent_factory.get_or_create_agent(workflow_config.supervisor_agent)
        if not supervisor_agent:
            logger.error(f"无法创建监督智能体: {workflow_config.supervisor_agent}")
            return None
        
        supervisor_class = supervisor_agent.__class__
        
        return self.register_workflow(
            name=workflow_name,
            supervisor_class=supervisor_class,
            config=getattr(workflow_config, 'parameters', {}),
            description=workflow_config.description,
            schedule_expression=getattr(workflow_config, 'schedule_expression', None)
        )
    
    def execute_workflow(self, 
                        workflow_name: str, 
                        task: str, 
                        **kwargs) -> WorkflowExecution:
        """执行指定工作流"""
        if workflow_name not in self.workflows:
            raise ValueError(f"工作流 '{workflow_name}' 未注册")
        
        registration = self.workflows[workflow_name]
        if not registration.enabled:
            raise ValueError(f"工作流 '{workflow_name}' 已禁用")
        
        execution_id = f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"开始执行工作流: {workflow_name} (ID: {execution_id})")
        
        execution = WorkflowExecution(
            workflow_name=workflow_name,
            execution_id=execution_id,
            start_time=start_time,
            metadata=kwargs
        )
        
        try:
            # 获取执行器
            executor = registration.executor
            if not executor:
                raise ValueError(f"工作流执行器未找到: {workflow_name}")
            
            # 执行任务
            result = executor.execute_single_task(
                task=task,
                agent_name=registration.config.get('agent_name', 'default_supervisor'),
                **kwargs
            )
            
            # 更新执行记录
            execution.end_time = datetime.now()
            execution.success = result.success
            execution.result = result
            if not result.success:
                execution.error = result.error
            
            # 更新统计
            registration.execution_count += 1
            registration.last_execution = execution.end_time
            if result.success:
                registration.success_count += 1
            else:
                registration.failure_count += 1
            
            logger.info(f"工作流执行完成: {workflow_name}, 成功: {result.success}")
            
        except Exception as e:
            execution.end_time = datetime.now()
            execution.success = False
            execution.error = str(e)
            
            registration.execution_count += 1
            registration.failure_count += 1
            registration.last_execution = execution.end_time
            
            logger.error(f"工作流执行失败: {workflow_name} - {e}")
        
        # 保存执行历史
        self.execution_history.append(execution)
        
        return execution
    
    def execute_batch_workflow(self, 
                              workflow_name: str, 
                              tasks: List[str], 
                              **kwargs) -> WorkflowExecution:
        """批量执行工作流"""
        if workflow_name not in self.workflows:
            raise ValueError(f"工作流 '{workflow_name}' 未注册")
        
        registration = self.workflows[workflow_name]
        if not registration.enabled:
            raise ValueError(f"工作流 '{workflow_name}' 已禁用")
        
        execution_id = f"{workflow_name}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"开始批量执行工作流: {workflow_name} (ID: {execution_id}), 任务数: {len(tasks)}")
        
        execution = WorkflowExecution(
            workflow_name=workflow_name,
            execution_id=execution_id,
            start_time=start_time,
            metadata={**kwargs, "batch_size": len(tasks)}
        )
        
        try:
            # 获取执行器
            executor = registration.executor
            if not executor:
                raise ValueError(f"工作流执行器未找到: {workflow_name}")
            
            # 批量执行任务
            batch_result = executor.execute_batch_tasks(
                tasks=tasks,
                agent_name=registration.config.get('agent_name', 'default_supervisor'),
                max_workers=registration.config.get('max_workers', 3),
                **kwargs
            )
            
            # 更新执行记录
            execution.end_time = datetime.now()
            execution.success = batch_result.success_rate > 0.5  # 成功率超过50%算成功
            execution.result = batch_result
            
            # 更新统计
            registration.execution_count += 1
            registration.last_execution = execution.end_time
            if execution.success:
                registration.success_count += 1
            else:
                registration.failure_count += 1
            
            logger.info(f"批量工作流执行完成: {workflow_name}, 成功率: {batch_result.success_rate:.2%}")
            
        except Exception as e:
            execution.end_time = datetime.now()
            execution.success = False
            execution.error = str(e)
            
            registration.execution_count += 1
            registration.failure_count += 1
            registration.last_execution = execution.end_time
            
            logger.error(f"批量工作流执行失败: {workflow_name} - {e}")
        
        # 保存执行历史
        self.execution_history.append(execution)
        
        return execution
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            logger.warning("调度器已经在运行")
            return
        
        self.scheduler_running = True
        
        def run_scheduler():
            logger.info("调度器启动")
            while self.scheduler_running:
                try:
                    schedule.run_pending()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"调度器执行错误: {e}")
        
        self.scheduler_thread = Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("工作流调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("工作流调度器已停止")
    
    def _add_to_scheduler(self, registration: WorkflowRegistration):
        """添加工作流到调度器"""
        if not registration.schedule_expression:
            return
        
        def job_wrapper():
            try:
                self.execute_workflow(
                    registration.name,
                    f"定时执行工作流: {registration.name}",
                    scheduled=True
                )
            except Exception as e:
                logger.error(f"定时工作流执行失败: {registration.name} - {e}")
        
        # 简化的调度表达式解析
        expr = registration.schedule_expression.lower()
        
        if "every" in expr and "minute" in expr:
            if "5" in expr:
                schedule.every(5).minutes.do(job_wrapper)
            elif "10" in expr:
                schedule.every(10).minutes.do(job_wrapper)
            elif "30" in expr:
                schedule.every(30).minutes.do(job_wrapper)
            else:
                schedule.every().minute.do(job_wrapper)
        elif "hourly" in expr or "hour" in expr:
            schedule.every().hour.do(job_wrapper)
        elif "daily" in expr or "day" in expr:
            schedule.every().day.do(job_wrapper)
        else:
            logger.warning(f"不支持的调度表达式: {registration.schedule_expression}")
        
        logger.info(f"工作流 {registration.name} 已添加到调度器: {registration.schedule_expression}")
    
    def get_workflow_status(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        if workflow_name not in self.workflows:
            return None
        
        registration = self.workflows[workflow_name]
        
        # 计算成功率
        success_rate = 0.0
        if registration.execution_count > 0:
            success_rate = registration.success_count / registration.execution_count
        
        return {
            "name": registration.name,
            "enabled": registration.enabled,
            "description": registration.description,
            "execution_count": registration.execution_count,
            "success_count": registration.success_count,
            "failure_count": registration.failure_count,
            "success_rate": success_rate,
            "last_execution": registration.last_execution.isoformat() if registration.last_execution else None,
            "schedule_expression": registration.schedule_expression,
            "has_executor": registration.executor is not None
        }
    
    def list_workflows(self) -> Dict[str, Dict[str, Any]]:
        """列出所有工作流"""
        result = {}
        for name in self.workflows.keys():
            status = self.get_workflow_status(name)
            if status is not None:
                result[name] = status
        return result
    
    def enable_workflow(self, workflow_name: str):
        """启用工作流"""
        if workflow_name in self.workflows:
            self.workflows[workflow_name].enabled = True
            logger.info(f"工作流已启用: {workflow_name}")
    
    def disable_workflow(self, workflow_name: str):
        """禁用工作流"""
        if workflow_name in self.workflows:
            self.workflows[workflow_name].enabled = False
            logger.info(f"工作流已禁用: {workflow_name}")
    
    def remove_workflow(self, workflow_name: str):
        """移除工作流"""
        if workflow_name in self.workflows:
            del self.workflows[workflow_name]
            logger.info(f"工作流已移除: {workflow_name}")
    
    def get_execution_history(self, 
                            workflow_name: Optional[str] = None, 
                            limit: int = 10) -> List[WorkflowExecution]:
        """获取执行历史"""
        history = self.execution_history
        
        if workflow_name:
            history = [ex for ex in history if ex.workflow_name == workflow_name]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x.start_time, reverse=True)
        
        return history[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        total_workflows = len(self.workflows)
        enabled_workflows = len([w for w in self.workflows.values() if w.enabled])
        total_executions = sum(w.execution_count for w in self.workflows.values())
        total_successes = sum(w.success_count for w in self.workflows.values())
        
        overall_success_rate = 0.0
        if total_executions > 0:
            overall_success_rate = total_successes / total_executions
        
        return {
            "total_workflows": total_workflows,
            "enabled_workflows": enabled_workflows,
            "disabled_workflows": total_workflows - enabled_workflows,
            "total_executions": total_executions,
            "total_successes": total_successes,
            "total_failures": total_executions - total_successes,
            "overall_success_rate": overall_success_rate,
            "scheduler_running": self.scheduler_running,
            "execution_history_count": len(self.execution_history)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """导出配置"""
        config = {
            "workflows": {},
            "statistics": self.get_statistics(),
            "export_time": datetime.now().isoformat()
        }
        
        for name, registration in self.workflows.items():
            config["workflows"][name] = {
                "name": registration.name,
                "config": registration.config,
                "enabled": registration.enabled,
                "description": registration.description,
                "schedule_expression": registration.schedule_expression,
                "execution_count": registration.execution_count,
                "success_count": registration.success_count,
                "failure_count": registration.failure_count
            }
        
        return config
    
    def import_configuration(self, config: Dict[str, Any]):
        """导入配置"""
        workflows_config = config.get("workflows", {})
        
        for name, workflow_config in workflows_config.items():
            try:
                self.register_workflow(
                    name=name,
                    supervisor_class=type,  # 占位符，实际需要从配置推断
                    config=workflow_config.get("config", {}),
                    description=workflow_config.get("description", ""),
                    schedule_expression=workflow_config.get("schedule_expression")
                )
                
                # 恢复统计信息
                if name in self.workflows:
                    registration = self.workflows[name]
                    registration.enabled = workflow_config.get("enabled", True)
                    registration.execution_count = workflow_config.get("execution_count", 0)
                    registration.success_count = workflow_config.get("success_count", 0)
                    registration.failure_count = workflow_config.get("failure_count", 0)
                
            except Exception as e:
                logger.error(f"导入工作流配置失败: {name} - {e}")
        
        logger.info(f"配置导入完成，导入 {len(workflows_config)} 个工作流")
    
    def cleanup_old_executions(self, days: int = 30):
        """清理旧的执行记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        original_count = len(self.execution_history)
        self.execution_history = [
            ex for ex in self.execution_history 
            if ex.start_time > cutoff_date
        ]
        
        cleaned_count = original_count - len(self.execution_history)
        logger.info(f"清理执行历史: 删除 {cleaned_count} 条记录 (保留 {days} 天内的记录)")
        
        return cleaned_count 