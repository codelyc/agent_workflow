"""
自检验证系统实现

对Agent输出进行多层次验证，包括结构验证、内容验证、AI自检等
"""

import json
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import xml.etree.ElementTree as ET

from .models import (
    ValidationRule, ValidationResult, ValidationStatus
)


class OutputValidator:
    """输出验证器"""
    
    def __init__(self, validation_rules: List[ValidationRule], llm_client=None, logger: Optional[logging.Logger] = None):
        self.validation_rules = validation_rules
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        # 按类型分组验证规则
        self.structure_rules = [r for r in validation_rules if r.type in ['schema', 'markup'] and r.enabled]
        self.content_rules = [r for r in validation_rules if r.type == 'content' and r.enabled]
        self.ai_check_rules = [r for r in validation_rules if r.type == 'ai_check' and r.enabled]
    
    async def validate_output(self, output: str, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """验证输出"""
        results = []
        context = context or {}
        
        self.logger.info(f"开始验证输出，长度: {len(output)}")
        
        # 结构验证
        for rule in self.structure_rules:
            result = await self._validate_structure(output, rule, context)
            results.append(result)
        
        # 内容验证
        for rule in self.content_rules:
            result = await self._validate_content(output, rule, context)
            results.append(result)
        
        # AI自检验证
        for rule in self.ai_check_rules:
            result = await self._ai_self_check(output, rule, context)
            results.append(result)
        
        # 统计结果
        failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        
        self.logger.info(f"验证完成: 失败 {failed_count}, 警告 {warning_count}, 总计 {len(results)}")
        
        return results
    
    async def _validate_structure(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> ValidationResult:
        """结构验证"""
        try:
            if rule.name == "json_schema_validator":
                return await self._validate_json_schema(output, rule)
            elif rule.name == "xml_validator":
                return await self._validate_xml(output, rule)
            else:
                return ValidationResult(
                    status=ValidationStatus.SKIPPED,
                    rule_name=rule.name,
                    message="未实现的结构验证"
                )
        except Exception as e:
            self.logger.error(f"结构验证异常 {rule.name}: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=rule.name,
                message=f"验证异常: {str(e)}"
            )
    
    async def _validate_json_schema(self, output: str, rule: ValidationRule) -> ValidationResult:
        """JSON Schema验证"""
        try:
            # 首先检查是否是有效JSON
            try:
                parsed_json = json.loads(output.strip())
            except json.JSONDecodeError as e:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=rule.name,
                    message=f"JSON格式无效: {str(e)}",
                    suggestions=["修正JSON语法错误"]
                )
            
            # 如果有schema文件，进行schema验证
            if rule.schema_file:
                schema_path = Path(rule.schema_file)
                if schema_path.exists():
                    try:
                        # 这里可以集成jsonschema库进行验证
                        # import jsonschema
                        # with open(schema_path) as f:
                        #     schema = json.load(f)
                        # jsonschema.validate(parsed_json, schema)
                        
                        # 简单示例验证
                        return ValidationResult(
                            status=ValidationStatus.PASSED,
                            rule_name=rule.name,
                            message="JSON结构验证通过"
                        )
                    except Exception as e:
                        return ValidationResult(
                            status=ValidationStatus.FAILED,
                            rule_name=rule.name,
                            message=f"Schema验证失败: {str(e)}",
                            suggestions=["检查JSON结构是否符合Schema要求"]
                        )
                else:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        rule_name=rule.name,
                        message=f"Schema文件不存在: {rule.schema_file}"
                    )
            else:
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=rule.name,
                    message="JSON格式验证通过"
                )
                
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=rule.name,
                message=f"JSON验证异常: {str(e)}"
            )
    
    async def _validate_xml(self, output: str, rule: ValidationRule) -> ValidationResult:
        """XML验证"""
        try:
            if rule.validate_well_formed:
                try:
                    ET.fromstring(output.strip())
                    return ValidationResult(
                        status=ValidationStatus.PASSED,
                        rule_name=rule.name,
                        message="XML格式验证通过"
                    )
                except ET.ParseError as e:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        rule_name=rule.name,
                        message=f"XML格式无效: {str(e)}",
                        suggestions=["修正XML语法错误", "检查标签是否正确闭合"]
                    )
            else:
                return ValidationResult(
                    status=ValidationStatus.SKIPPED,
                    rule_name=rule.name,
                    message="XML验证已禁用"
                )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=rule.name,
                message=f"XML验证异常: {str(e)}"
            )
    
    async def _validate_content(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> ValidationResult:
        """内容验证"""
        try:
            issues = []
            suggestions = []
            
            # 拼写检查
            if rule.spell_check:
                spell_issues = await self._check_spelling(output)
                if spell_issues:
                    issues.extend([f"拼写错误: {issue}" for issue in spell_issues])
                    suggestions.append("修正拼写错误")
            
            # 语法检查
            if rule.grammar_check:
                grammar_issues = await self._check_grammar(output)
                if grammar_issues:
                    issues.extend([f"语法错误: {issue}" for issue in grammar_issues])
                    suggestions.append("修正语法错误")
            
            # 事实检查
            if rule.fact_check:
                fact_issues = await self._check_facts(output, context)
                if fact_issues:
                    issues.extend([f"事实错误: {issue}" for issue in fact_issues])
                    suggestions.append("验证并修正事实错误")
            
            if issues:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=rule.name,
                    message="; ".join(issues),
                    suggestions=suggestions
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=rule.name,
                    message="内容验证通过"
                )
                
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=rule.name,
                message=f"内容验证异常: {str(e)}"
            )
    
    async def _check_spelling(self, output: str) -> List[str]:
        """拼写检查（简单实现）"""
        # 这里可以集成拼写检查库，如pyspellchecker
        issues = []
        
        # 简单的拼写检查示例
        common_errors = {
            'teh': 'the',
            'recieve': 'receive',
            'occured': 'occurred',
            'seperate': 'separate'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', output.lower())
        for word in words:
            if word in common_errors:
                issues.append(f"'{word}' 应为 '{common_errors[word]}'")
        
        return issues
    
    async def _check_grammar(self, output: str) -> List[str]:
        """语法检查（简单实现）"""
        # 这里可以集成语法检查库，如language-tool-python
        issues = []
        
        # 简单的语法检查示例
        lines = output.split('\n')
        for i, line in enumerate(lines, 1):
            # 检查句子是否以标点结尾
            if line.strip() and not line.strip()[-1] in '.!?':
                if len(line.strip()) > 20:  # 只对较长的句子检查
                    issues.append(f"第{i}行: 句子缺少结尾标点")
        
        return issues
    
    async def _check_facts(self, output: str, context: Dict[str, Any]) -> List[str]:
        """事实检查（简单实现）"""
        # 这里可以集成事实检查服务或知识图谱
        issues = []
        
        # 简单的事实检查示例
        # 检查一些明显的事实错误
        fact_patterns = [
            (r'地球是平的', '地球是球形的'),
            (r'太阳围绕地球转', '地球围绕太阳转'),
        ]
        
        for pattern, correction in fact_patterns:
            if re.search(pattern, output):
                issues.append(f"事实错误: {correction}")
        
        return issues
    
    async def _ai_self_check(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> ValidationResult:
        """AI自检验证"""
        if not self.llm_client:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                rule_name=rule.name,
                message="LLM客户端未配置"
            )
        
        try:
            results = []
            
            # 一致性检查
            if 'consistency' in rule.check_prompts:
                consistency_result = await self._ai_consistency_check(output, rule, context)
                results.append(consistency_result)
            
            # 格式检查
            if 'format_check' in rule.check_prompts:
                format_result = await self._ai_format_check(output, rule, context)
                results.append(format_result)
            
            # 质量检查
            if 'quality_check' in rule.check_prompts:
                quality_result = await self._ai_quality_check(output, rule, context)
                results.append(quality_result)
            
            # 汇总结果
            failed_checks = [r for r in results if not r.get('success', True)]
            if failed_checks:
                issues = []
                suggestions = []
                
                for check in failed_checks:
                    issues.extend(check.get('issues', []))
                    suggestions.extend(check.get('suggestions', []))
                
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    rule_name=rule.name,
                    message=f"AI自检发现问题: {'; '.join(issues)}",
                    suggestions=suggestions,
                    details={'ai_check_results': results}
                )
            else:
                # 计算平均分
                scores = [r.get('score', 0) for r in results if 'score' in r]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    rule_name=rule.name,
                    message="AI自检通过",
                    score=avg_score,
                    details={'ai_check_results': results}
                )
                
        except Exception as e:
            self.logger.error(f"AI自检异常 {rule.name}: {e}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=rule.name,
                message=f"AI自检异常: {str(e)}"
            )
    
    async def _ai_consistency_check(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI一致性检查"""
        prompt = rule.check_prompts['consistency'].format(output=output)
        
        try:
            response = await self._call_llm(prompt, rule.model)
            result = json.loads(response)
            
            return {
                'type': 'consistency',
                'success': result.get('consistent', False) and result.get('complete', False),
                'score': result.get('score', 0),
                'issues': result.get('issues', []),
                'suggestions': ['修正逻辑矛盾', '补充缺失信息'] if not result.get('consistent', True) else []
            }
        except Exception as e:
            return {
                'type': 'consistency',
                'success': False,
                'issues': [f'一致性检查失败: {str(e)}']
            }
    
    async def _ai_format_check(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI格式检查"""
        format_requirements = context.get('format_requirements', '未指定')
        prompt = rule.check_prompts['format_check'].format(
            output=output,
            format_requirements=format_requirements
        )
        
        try:
            response = await self._call_llm(prompt, rule.model)
            result = json.loads(response)
            
            return {
                'type': 'format',
                'success': result.get('format_valid', False),
                'issues': result.get('issues', []),
                'suggestions': ['调整输出格式'] if not result.get('format_valid', True) else []
            }
        except Exception as e:
            return {
                'type': 'format',
                'success': False,
                'issues': [f'格式检查失败: {str(e)}']
            }
    
    async def _ai_quality_check(self, output: str, rule: ValidationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI质量检查"""
        prompt = rule.check_prompts['quality_check'].format(output=output)
        
        try:
            response = await self._call_llm(prompt, rule.model)
            result = json.loads(response)
            
            overall_score = result.get('overall', 0)
            success = overall_score >= 7  # 7分以上认为通过
            
            return {
                'type': 'quality',
                'success': success,
                'score': overall_score,
                'accuracy': result.get('accuracy', 0),
                'completeness': result.get('completeness', 0),
                'clarity': result.get('clarity', 0),
                'usefulness': result.get('usefulness', 0),
                'feedback': result.get('feedback', ''),
                'suggestions': [result.get('feedback', '')] if not success else []
            }
        except Exception as e:
            return {
                'type': 'quality',
                'success': False,
                'issues': [f'质量检查失败: {str(e)}']
            }
    
    async def _call_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """调用LLM"""
        if not self.llm_client:
            raise ValueError("LLM客户端未配置")
        
        # 这里需要实现具体的LLM调用逻辑
        # 根据你的LLM客户端实现来调整
        try:
            if hasattr(self.llm_client, 'chat_completions_create'):
                response = await self.llm_client.chat_completions_create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            elif hasattr(self.llm_client, 'generate'):
                response = await self.llm_client.generate(prompt, model=model)
                return response
            else:
                raise ValueError("不支持的LLM客户端")
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """获取验证摘要"""
        total_count = len(results)
        passed_count = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped_count = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        
        # 计算平均分
        scores = [r.score for r in results if r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        
        # 收集所有建议
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.suggestions)
        
        return {
            'total_count': total_count,
            'passed_count': passed_count,
            'failed_count': failed_count,
            'warning_count': warning_count,
            'skipped_count': skipped_count,
            'success_rate': passed_count / total_count if total_count > 0 else 0,
            'average_score': avg_score,
            'has_failures': failed_count > 0,
            'suggestions': list(set(all_suggestions)),  # 去重
            'failed_rules': [r.rule_name for r in results if r.status == ValidationStatus.FAILED]
        }


class ValidationPipeline:
    """验证管道"""
    
    def __init__(self, validators: List[OutputValidator]):
        self.validators = validators
        self.logger = logging.getLogger(__name__)
    
    async def run_validation(self, output: str, context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """运行验证管道"""
        all_results = []
        
        for validator in self.validators:
            try:
                results = await validator.validate_output(output, context)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"验证器运行失败: {e}")
                # 继续运行其他验证器
        
        return all_results 