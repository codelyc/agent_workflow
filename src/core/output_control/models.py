"""
输出控制系统的数据模型定义
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import yaml
import json
from pathlib import Path


class ConstraintType(Enum):
    """约束类型枚举"""
    FORMAT = "format"
    CONTENT = "content" 
    LENGTH = "length"
    LOGIC = "logic"
    SCHEMA = "schema"


class ValidationStatus(Enum):
    """验证状态枚举"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ProcessingStatus(Enum):
    """处理状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class RePromptStrategy(Enum):
    """重新提示策略枚举"""
    PROGRESSIVE = "progressive"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class Constraint:
    """约束定义"""
    name: str
    description: str
    type: ConstraintType
    required: bool = True
    enabled: bool = True
    
    # 格式约束特有属性
    pattern: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    # 长度约束特有属性
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # 内容约束特有属性
    blacklist_patterns: List[str] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)
    optional_elements: List[str] = field(default_factory=list)
    case_sensitive: bool = False
    
    # 逻辑约束特有属性
    check_contradictions: bool = False
    check_completeness: bool = False
    verify_facts: bool = False
    cross_reference: bool = False


@dataclass
class ValidationRule:
    """验证规则定义"""
    name: str
    type: str
    enabled: bool = True
    description: str = ""
    
    # Schema验证
    schema_file: Optional[str] = None
    validate_well_formed: bool = False
    
    # 内容验证
    spell_check: bool = False
    grammar_check: bool = False
    fact_check: bool = False
    
    # AI自检
    model: Optional[str] = None
    check_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessingRule:
    """后处理规则定义"""
    name: str
    type: str
    enabled: bool = True
    description: str = ""
    
    # 文本处理
    strip_whitespace: bool = False
    normalize_line_endings: bool = False
    remove_empty_lines: bool = False
    
    # 格式化
    auto_format_json: bool = False
    auto_format_markdown: bool = False
    prettify_code_blocks: bool = False
    
    # 补全
    add_missing_sections: bool = False
    generate_summary: bool = False
    add_metadata: bool = False
    
    # 过滤
    remove_sensitive_info: bool = False
    filter_inappropriate_content: bool = False


@dataclass
class RePromptTemplate:
    """重新提示模板"""
    name: str
    template: str
    description: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    status: ValidationStatus
    rule_name: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """处理结果"""
    status: ProcessingStatus
    rule_name: str
    processed_content: str = ""
    original_content: str = ""
    changes_made: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ControlResult:
    """整体控制结果"""
    success: bool
    final_output: str
    original_output: str = ""
    
    # 各阶段结果
    constraint_violations: List[ValidationResult] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    processing_results: List[ProcessingResult] = field(default_factory=list)
    
    # 重新提示信息
    re_prompt_attempts: int = 0
    re_prompt_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    processing_time: float = 0.0
    timestamp: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputControlConfig:
    """输出控制配置"""
    
    # 全局设置
    enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: int = 30
    debug_mode: bool = False
    save_intermediate_results: bool = True
    
    # 约束、验证、处理规则
    pre_constraints: List[Constraint] = field(default_factory=list)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    processing_rules: List[ProcessingRule] = field(default_factory=list)
    
    # Re-prompt 配置
    re_prompt_enabled: bool = True
    re_prompt_max_attempts: int = 3
    re_prompt_strategy: RePromptStrategy = RePromptStrategy.PROGRESSIVE
    re_prompt_templates: Dict[str, RePromptTemplate] = field(default_factory=dict)
    
    # 奖励模型配置
    reward_model_enabled: bool = False
    reward_model_type: str = "classification"
    reward_threshold: float = 0.7
    evaluation_criteria: Dict[str, float] = field(default_factory=dict)
    
    # 输出类型特定配置
    output_type_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 日志配置
    logging_enabled: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_validation_details: bool = True
    log_processing_steps: bool = True
    log_re_prompt_attempts: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'OutputControlConfig':
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputControlConfig':
        """从字典创建配置"""
        # 解析全局设置
        global_config = data.get('global', {})
        
        # 解析前置约束
        pre_constraints = []
        pre_constraints_data = data.get('pre_constraints', {})
        
        # 解析格式约束
        for constraint_data in pre_constraints_data.get('format_constraints', []):
            constraint = Constraint(
                name=constraint_data['name'],
                description=constraint_data['description'],
                type=ConstraintType.FORMAT,
                required=constraint_data.get('required', True),
                pattern=constraint_data.get('pattern'),
                examples=constraint_data.get('examples', [])
            )
            pre_constraints.append(constraint)
            
        # 解析内容约束
        for constraint_data in pre_constraints_data.get('content_constraints', []):
            constraint = Constraint(
                name=constraint_data['name'],
                description=constraint_data['description'],
                type=ConstraintType.CONTENT,
                required=constraint_data.get('required', True),
                blacklist_patterns=constraint_data.get('blacklist_patterns', []),
                required_elements=constraint_data.get('required_elements', []),
                optional_elements=constraint_data.get('optional_elements', []),
                case_sensitive=constraint_data.get('case_sensitive', False)
            )
            pre_constraints.append(constraint)
            
        # 解析长度约束
        for constraint_data in pre_constraints_data.get('length_constraints', []):
            constraint = Constraint(
                name=constraint_data['name'],
                description=constraint_data['description'],
                type=ConstraintType.LENGTH,
                required=constraint_data.get('required', True),
                min_length=constraint_data.get('min_length'),
                max_length=constraint_data.get('max_length')
            )
            pre_constraints.append(constraint)
            
        # 解析逻辑约束
        for constraint_data in pre_constraints_data.get('logic_constraints', []):
            constraint = Constraint(
                name=constraint_data['name'],
                description=constraint_data['description'],
                type=ConstraintType.LOGIC,
                required=constraint_data.get('required', True),
                check_contradictions=constraint_data.get('check_contradictions', False),
                check_completeness=constraint_data.get('check_completeness', False),
                verify_facts=constraint_data.get('verify_facts', False),
                cross_reference=constraint_data.get('cross_reference', False)
            )
            pre_constraints.append(constraint)
        
        # 解析验证规则
        validation_rules = []
        validation_data = data.get('validation', {})
        
        # 结构验证
        struct_validation = validation_data.get('structure_validation', {})
        if struct_validation.get('enabled', False):
            for validator_data in struct_validation.get('custom_validators', []):
                rule = ValidationRule(
                    name=validator_data['name'],
                    type=validator_data['type'],
                    schema_file=validator_data.get('schema_file'),
                    validate_well_formed=validator_data.get('validate_well_formed', False)
                )
                validation_rules.append(rule)
        
        # 内容验证
        content_validation = validation_data.get('content_validation', {})
        if content_validation.get('enabled', False):
            rule = ValidationRule(
                name="content_validation",
                type="content",
                spell_check=content_validation.get('spell_check', False),
                grammar_check=content_validation.get('grammar_check', False),
                fact_check=content_validation.get('fact_check', False)
            )
            validation_rules.append(rule)
            
        # AI自检验证
        ai_check = validation_data.get('ai_self_check', {})
        if ai_check.get('enabled', False):
            rule = ValidationRule(
                name="ai_self_check",
                type="ai_check",
                model=ai_check.get('model'),
                check_prompts=ai_check.get('check_prompts', {})
            )
            validation_rules.append(rule)
        
        # 解析后处理规则
        processing_rules = []
        post_processing_data = data.get('post_processing', {})
        
        # 文本处理
        text_processing = post_processing_data.get('text_processing', {})
        if text_processing.get('enabled', False):
            rule = ProcessingRule(
                name="text_processing",
                type="text",
                strip_whitespace=text_processing.get('strip_whitespace', False),
                normalize_line_endings=text_processing.get('normalize_line_endings', False),
                remove_empty_lines=text_processing.get('remove_empty_lines', False)
            )
            processing_rules.append(rule)
            
        # 格式化处理
        formatting = post_processing_data.get('formatting', {})
        if formatting.get('enabled', False):
            rule = ProcessingRule(
                name="formatting",
                type="format",
                auto_format_json=formatting.get('auto_format_json', False),
                auto_format_markdown=formatting.get('auto_format_markdown', False),
                prettify_code_blocks=formatting.get('prettify_code_blocks', False)
            )
            processing_rules.append(rule)
            
        # 补全处理
        completion = post_processing_data.get('completion', {})
        if completion.get('enabled', False):
            rule = ProcessingRule(
                name="completion",
                type="completion",
                add_missing_sections=completion.get('add_missing_sections', False),
                generate_summary=completion.get('generate_summary', False),
                add_metadata=completion.get('add_metadata', False)
            )
            processing_rules.append(rule)
            
        # 过滤处理
        filtering = post_processing_data.get('filtering', {})
        if filtering.get('enabled', False):
            rule = ProcessingRule(
                name="filtering",
                type="filter",
                remove_sensitive_info=filtering.get('remove_sensitive_info', False),
                filter_inappropriate_content=filtering.get('filter_inappropriate_content', False)
            )
            processing_rules.append(rule)
        
        # 解析re-prompt配置
        re_prompt_data = data.get('re_prompt', {})
        re_prompt_templates = {}
        for name, template_text in re_prompt_data.get('templates', {}).items():
            re_prompt_templates[name] = RePromptTemplate(
                name=name,
                template=template_text,
                description=f"Re-prompt template for {name}"
            )
        
        # 解析奖励模型配置
        reward_model_data = data.get('reward_model', {})
        evaluation_criteria = {}
        for criteria in reward_model_data.get('evaluation_criteria', []):
            evaluation_criteria[criteria['name']] = criteria['weight']
        
        # 解析日志配置
        logging_data = data.get('logging', {})
        
        return cls(
            enabled=global_config.get('enabled', True),
            max_retry_attempts=global_config.get('max_retry_attempts', 3),
            timeout_seconds=global_config.get('timeout_seconds', 30),
            debug_mode=global_config.get('debug_mode', False),
            save_intermediate_results=global_config.get('save_intermediate_results', True),
            
            pre_constraints=pre_constraints,
            validation_rules=validation_rules,
            processing_rules=processing_rules,
            
            re_prompt_enabled=re_prompt_data.get('enabled', True),
            re_prompt_max_attempts=re_prompt_data.get('max_attempts', 3),
            re_prompt_strategy=RePromptStrategy(re_prompt_data.get('strategy', 'progressive')),
            re_prompt_templates=re_prompt_templates,
            
            reward_model_enabled=reward_model_data.get('enabled', False),
            reward_model_type=reward_model_data.get('model_type', 'classification'),
            reward_threshold=reward_model_data.get('threshold', 0.7),
            evaluation_criteria=evaluation_criteria,
            
            output_type_configs=data.get('output_types', {}),
            
            logging_enabled=logging_data.get('enabled', True),
            log_level=logging_data.get('level', 'INFO'),
            log_file=logging_data.get('log_file'),
            log_validation_details=logging_data.get('log_validation_details', True),
            log_processing_steps=logging_data.get('log_processing_steps', True),
            log_re_prompt_attempts=logging_data.get('log_re_prompt_attempts', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 这里可以实现配置的序列化
        pass
    
    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """保存配置到YAML文件"""
        # 这里可以实现配置的保存
        pass 