"""
前置约束系统实现

在Agent输出前应用各种约束条件，确保输出符合预期格式和内容要求
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from .models import (
    Constraint, ConstraintType, ValidationResult, ValidationStatus
)


class ConstraintEngine:
    """约束引擎"""
    
    def __init__(self, constraints: List[Constraint], logger: Optional[logging.Logger] = None):
        self.constraints = constraints
        self.logger = logger or logging.getLogger(__name__)
        
        # 按类型分组约束
        self.format_constraints = [c for c in constraints if c.type == ConstraintType.FORMAT and c.enabled]
        self.content_constraints = [c for c in constraints if c.type == ConstraintType.CONTENT and c.enabled]
        self.length_constraints = [c for c in constraints if c.type == ConstraintType.LENGTH and c.enabled]
        self.logic_constraints = [c for c in constraints if c.type == ConstraintType.LOGIC and c.enabled]
    
    def check_constraints(self, output: str, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """检查所有约束"""
        results = []
        context = context or {}
        
        self.logger.info(f"开始检查约束，输出长度: {len(output)}")
        
        # 检查格式约束
        for constraint in self.format_constraints:
            result = self._check_format_constraint(output, constraint, context)
            results.append(result)
            if result.status == ValidationStatus.FAILED and constraint.required:
                self.logger.warning(f"必需约束失败: {constraint.name}")
        
        # 检查内容约束
        for constraint in self.content_constraints:
            result = self._check_content_constraint(output, constraint, context)
            results.append(result)
            if result.status == ValidationStatus.FAILED and constraint.required:
                self.logger.warning(f"必需约束失败: {constraint.name}")
        
        # 检查长度约束
        for constraint in self.length_constraints:
            result = self._check_length_constraint(output, constraint, context)
            results.append(result)
            if result.status == ValidationStatus.FAILED and constraint.required:
                self.logger.warning(f"必需约束失败: {constraint.name}")
        
        # 检查逻辑约束
        for constraint in self.logic_constraints:
            result = self._check_logic_constraint(output, constraint, context)
            results.append(result)
            if result.status == ValidationStatus.FAILED and constraint.required:
                self.logger.warning(f"必需约束失败: {constraint.name}")
        
        # 统计结果
        failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        
        self.logger.info(f"约束检查完成: 失败 {failed_count}, 警告 {warning_count}, 总计 {len(results)}")
        
        return results
    
    def _check_format_constraint(self, output: str, constraint: Constraint, context: Dict[str, Any]) -> ValidationResult:
        """检查格式约束"""
        try:
            if constraint.name == "json_format":
                return self._check_json_format(output, constraint)
            elif constraint.name == "markdown_format":
                return self._check_markdown_format(output, constraint)
            elif constraint.pattern:
                return self._check_pattern_format(output, constraint)
            else:
                return ValidationResult(
                    status=ValidationStatus.SKIPPED,
                    rule_name=constraint.name,
                    message="未实现的格式约束"
                )
        except Exception as e:
            self.logger.error(f"格式约束检查异常 {constraint.name}: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"检查异常: {str(e)}"
            )
    
    def _check_json_format(self, output: str, constraint: Constraint) -> ValidationResult:
        """检查JSON格式"""
        try:
            # 尝试解析JSON
            json.loads(output.strip())
            return ValidationResult(
                status=ValidationStatus.PASSED,
                rule_name=constraint.name,
                message="JSON格式有效"
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"JSON格式无效: {str(e)}",
                suggestions=[
                    "确保JSON语法正确",
                    "检查括号、引号是否匹配",
                    "参考示例: " + str(constraint.examples)
                ]
            )
    
    def _check_markdown_format(self, output: str, constraint: Constraint) -> ValidationResult:
        """检查Markdown格式"""
        lines = output.strip().split('\n')
        if not lines:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message="输出为空"
            )
        
        # 检查是否包含Markdown元素
        has_headers = any(line.startswith('#') for line in lines)
        has_lists = any(line.strip().startswith(('*', '-', '+')) or re.match(r'^\d+\.', line.strip()) for line in lines)
        has_code_blocks = '```' in output
        has_emphasis = any(('*' in line or '_' in line) for line in lines)
        
        if has_headers or has_lists or has_code_blocks or has_emphasis:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                rule_name=constraint.name,
                message="包含Markdown格式元素"
            )
        else:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                rule_name=constraint.name,
                message="未检测到明显的Markdown格式元素",
                suggestions=[
                    "添加标题 (# ## ###)",
                    "使用列表 (* - +)",
                    "添加代码块 (```)",
                    "参考示例: " + str(constraint.examples)
                ]
            )
    
    def _check_pattern_format(self, output: str, constraint: Constraint) -> ValidationResult:
        """检查正则表达式模式"""
        try:
            if re.search(constraint.pattern, output, re.MULTILINE | re.DOTALL):
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=constraint.name,
                    message="匹配格式模式"
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=constraint.name,
                    message=f"不匹配格式模式: {constraint.pattern}",
                    suggestions=[
                        f"确保输出符合模式: {constraint.pattern}",
                        "参考示例: " + str(constraint.examples)
                    ]
                )
        except re.error as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"正则表达式错误: {str(e)}"
            )
    
    def _check_content_constraint(self, output: str, constraint: Constraint, context: Dict[str, Any]) -> ValidationResult:
        """检查内容约束"""
        try:
            if constraint.name == "no_sensitive_info":
                return self._check_sensitive_info(output, constraint)
            elif constraint.name == "required_elements":
                return self._check_required_elements(output, constraint)
            else:
                return ValidationResult(
                    status=ValidationStatus.SKIPPED,
                    rule_name=constraint.name,
                    message="未实现的内容约束"
                )
        except Exception as e:
            self.logger.error(f"内容约束检查异常 {constraint.name}: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"检查异常: {str(e)}"
            )
    
    def _check_sensitive_info(self, output: str, constraint: Constraint) -> ValidationResult:
        """检查敏感信息"""
        violations = []
        check_text = output.lower() if not constraint.case_sensitive else output
        
        for pattern in constraint.blacklist_patterns:
            check_pattern = pattern.lower() if not constraint.case_sensitive else pattern
            if check_pattern in check_text:
                violations.append(pattern)
        
        if violations:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"包含敏感信息: {', '.join(violations)}",
                suggestions=[
                    "移除或替换敏感信息",
                    "使用占位符代替真实值",
                    "确保不泄露机密数据"
                ]
            )
        else:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                rule_name=constraint.name,
                message="未检测到敏感信息"
            )
    
    def _check_required_elements(self, output: str, constraint: Constraint) -> ValidationResult:
        """检查必需元素"""
        missing_required = []
        missing_optional = []
        
        check_text = output.lower() if not constraint.case_sensitive else output
        
        # 检查必需元素
        for element in constraint.required_elements:
            check_element = element.lower() if not constraint.case_sensitive else element
            if check_element not in check_text:
                missing_required.append(element)
        
        # 检查可选元素
        for element in constraint.optional_elements:
            check_element = element.lower() if not constraint.case_sensitive else element
            if check_element not in check_text:
                missing_optional.append(element)
        
        if missing_required:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"缺少必需元素: {', '.join(missing_required)}",
                suggestions=[
                    f"添加缺少的必需元素: {', '.join(missing_required)}",
                    f"可选但推荐添加: {', '.join(missing_optional)}" if missing_optional else ""
                ]
            )
        elif missing_optional:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                rule_name=constraint.name,
                message=f"缺少可选元素: {', '.join(missing_optional)}",
                suggestions=[f"建议添加: {', '.join(missing_optional)}"]
            )
        else:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                rule_name=constraint.name,
                message="包含所有必需和可选元素"
            )
    
    def _check_length_constraint(self, output: str, constraint: Constraint, context: Dict[str, Any]) -> ValidationResult:
        """检查长度约束"""
        try:
            length = len(output)
            violations = []
            
            if constraint.min_length is not None and length < constraint.min_length:
                violations.append(f"长度 {length} 小于最小值 {constraint.min_length}")
            
            if constraint.max_length is not None and length > constraint.max_length:
                violations.append(f"长度 {length} 超过最大值 {constraint.max_length}")
            
            if violations:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=constraint.name,
                    message="; ".join(violations),
                    suggestions=[
                        f"调整输出长度至 {constraint.min_length}-{constraint.max_length} 字符范围内" if constraint.min_length and constraint.max_length else
                        f"增加内容至至少 {constraint.min_length} 字符" if constraint.min_length else
                        f"缩减内容至最多 {constraint.max_length} 字符"
                    ]
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=constraint.name,
                    message=f"长度 {length} 符合要求"
                )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"长度检查异常: {str(e)}"
            )
    
    def _check_logic_constraint(self, output: str, constraint: Constraint, context: Dict[str, Any]) -> ValidationResult:
        """检查逻辑约束"""
        try:
            issues = []
            
            if constraint.check_contradictions:
                contradictions = self._detect_contradictions(output)
                if contradictions:
                    issues.extend([f"逻辑矛盾: {c}" for c in contradictions])
            
            if constraint.check_completeness:
                completeness_issues = self._check_completeness(output, context)
                if completeness_issues:
                    issues.extend([f"完整性问题: {c}" for c in completeness_issues])
            
            if issues:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=constraint.name,
                    message="; ".join(issues),
                    suggestions=[
                        "检查并解决逻辑矛盾",
                        "补充缺失的信息",
                        "确保论证完整"
                    ]
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=constraint.name,
                    message="逻辑检查通过"
                )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message=f"逻辑检查异常: {str(e)}"
            )
    
    def _detect_contradictions(self, output: str) -> List[str]:
        """检测逻辑矛盾（简单实现）"""
        contradictions = []
        
        # 检测一些简单的矛盾模式
        lines = output.lower().split('\n')
        
        # 检测Yes/No矛盾
        has_yes = any('yes' in line or '是' in line or '正确' in line for line in lines)
        has_no = any('no' in line or '不' in line or '错误' in line for line in lines)
        
        if has_yes and has_no:
            # 进一步检查是否真的矛盾
            yes_contexts = [line for line in lines if 'yes' in line or '是' in line or '正确' in line]
            no_contexts = [line for line in lines if 'no' in line or '不' in line or '错误' in line]
            
            # 简单的矛盾检测逻辑
            if len(yes_contexts) > 0 and len(no_contexts) > 0:
                contradictions.append("同时包含肯定和否定表述")
        
        return contradictions
    
    def _check_completeness(self, output: str, context: Dict[str, Any]) -> List[str]:
        """检查完整性（简单实现）"""
        issues = []
        
        # 检查是否有结论
        if not any(keyword in output.lower() for keyword in ['结论', 'conclusion', '总结', 'summary']):
            issues.append("缺少明确结论")
        
        # 检查是否有推理过程
        if not any(keyword in output.lower() for keyword in ['因为', 'because', '由于', '所以', 'therefore']):
            issues.append("缺少推理过程")
        
        return issues
    
    def get_constraint_prompt_additions(self, output_type: Optional[str] = None) -> str:
        """获取约束条件的Prompt补充"""
        prompt_parts = []
        
        if self.format_constraints:
            prompt_parts.append("格式要求:")
            for constraint in self.format_constraints:
                if constraint.required:
                    prompt_parts.append(f"- {constraint.description}")
                    if constraint.examples:
                        prompt_parts.append(f"  示例: {constraint.examples[0]}")
        
        if self.content_constraints:
            prompt_parts.append("\n内容要求:")
            for constraint in self.content_constraints:
                if constraint.required:
                    prompt_parts.append(f"- {constraint.description}")
        
        if self.length_constraints:
            prompt_parts.append("\n长度要求:")
            for constraint in self.length_constraints:
                if constraint.required:
                    range_desc = ""
                    if constraint.min_length and constraint.max_length:
                        range_desc = f"{constraint.min_length}-{constraint.max_length}字符"
                    elif constraint.min_length:
                        range_desc = f"至少{constraint.min_length}字符"
                    elif constraint.max_length:
                        range_desc = f"最多{constraint.max_length}字符"
                    
                    if range_desc:
                        prompt_parts.append(f"- {constraint.description}: {range_desc}")
        
        if self.logic_constraints:
            prompt_parts.append("\n逻辑要求:")
            for constraint in self.logic_constraints:
                if constraint.required:
                    prompt_parts.append(f"- {constraint.description}")
        
        return "\n".join(prompt_parts) if prompt_parts else ""


class ConstraintViolationException(Exception):
    """约束违反异常"""
    
    def __init__(self, violations: List[ValidationResult]):
        self.violations = violations
        super().__init__(f"约束违反: {len(violations)} 个失败") 