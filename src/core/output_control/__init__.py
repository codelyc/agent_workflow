"""
Agent 输出控制系统

基于"前置约束+自检+后处理"思路设计的输出控制框架
"""

from .controller import OutputController
from .models import (
    OutputControlConfig,
    ValidationResult,
    ProcessingResult,
    ControlResult
)

__all__ = [
    'OutputController',
    'OutputControlConfig', 
    'ValidationResult',
    'ProcessingResult',
    'ControlResult'
]

__version__ = "1.0.0" 