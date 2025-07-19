"""
工作流自动化模块

包含自动化工作流管理器和调度功能
"""

from .workflow_manager import AutomatedWorkflowManager, WorkflowRegistration, WorkflowExecution

__all__ = [
    'AutomatedWorkflowManager',
    'WorkflowRegistration',
    'WorkflowExecution'
] 
 
 