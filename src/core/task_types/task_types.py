"""
任务类型定义

定义AI Agents系统中支持的各种任务类型
"""

from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    
    # 分析型任务
    ANALYSIS = "analysis"
    REPO_ANALYSIS = "repo_analysis"
    CODE_ANALYSIS = "code_analysis"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    
    # 生成型任务
    CODE_GENERATION = "code_generation"
    DOCUMENTATION = "documentation"
    TEST_GENERATION = "test_generation"
    
    # 优化型任务
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    PERFORMANCE_TUNING = "performance_tuning"
    
    # 搜索型任务
    SEARCH = "search"
    FILE_SEARCH = "file_search"
    CODE_SEARCH = "code_search"
    
    # 执行型任务
    EXECUTION = "execution"
    TEST_EXECUTION = "test_execution"
    BUILD = "build"
    DEPLOYMENT = "deployment"
    
    # 验证型任务
    VALIDATION = "validation"
    QUALITY_CHECK = "quality_check"
    SECURITY_CHECK = "security_check"
    
    # 自定义任务
    CUSTOM = "custom"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, task_type_str: str):
        """从字符串创建任务类型"""
        for task_type in cls:
            if task_type.value == task_type_str.lower():
                return task_type
        return cls.CUSTOM
    
    @property
    def description(self) -> str:
        """获取任务类型描述"""
        descriptions = {
            self.ANALYSIS: "数据分析和代码审查类任务",
            self.REPO_ANALYSIS: "代码仓库分析任务",
            self.CODE_ANALYSIS: "代码质量和结构分析",
            self.ARCHITECTURE_ANALYSIS: "架构设计分析",
            
            self.CODE_GENERATION: "代码生成和创建任务",
            self.DOCUMENTATION: "文档生成和维护",
            self.TEST_GENERATION: "测试用例生成",
            
            self.OPTIMIZATION: "性能和质量优化任务",
            self.REFACTORING: "代码重构和改进",
            self.PERFORMANCE_TUNING: "性能调优",
            
            self.SEARCH: "通用搜索任务",
            self.FILE_SEARCH: "文件搜索和定位",
            self.CODE_SEARCH: "代码搜索和查找",
            
            self.EXECUTION: "任务执行和处理",
            self.TEST_EXECUTION: "测试执行和验证",
            self.BUILD: "构建和编译任务",
            self.DEPLOYMENT: "部署和发布任务",
            
            self.VALIDATION: "验证和检查任务",
            self.QUALITY_CHECK: "质量检查和评估",
            self.SECURITY_CHECK: "安全检查和审计",
            
            self.CUSTOM: "自定义任务类型"
        }
        return descriptions.get(self, "未知任务类型")
    
    @property
    def category(self) -> str:
        """获取任务类别"""
        if self in [self.ANALYSIS, self.REPO_ANALYSIS, self.CODE_ANALYSIS, self.ARCHITECTURE_ANALYSIS]:
            return "分析"
        elif self in [self.CODE_GENERATION, self.DOCUMENTATION, self.TEST_GENERATION]:
            return "生成"
        elif self in [self.OPTIMIZATION, self.REFACTORING, self.PERFORMANCE_TUNING]:
            return "优化"
        elif self in [self.SEARCH, self.FILE_SEARCH, self.CODE_SEARCH]:
            return "搜索"
        elif self in [self.EXECUTION, self.TEST_EXECUTION, self.BUILD, self.DEPLOYMENT]:
            return "执行"
        elif self in [self.VALIDATION, self.QUALITY_CHECK, self.SECURITY_CHECK]:
            return "验证"
        else:
            return "自定义" 