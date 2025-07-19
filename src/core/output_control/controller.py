"""
输出控制主控制器

整合前置约束、自检验证、后处理和re-prompt功能，实现完整的Agent输出控制流程
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from .models import (
    OutputControlConfig, ControlResult, ValidationResult, ProcessingResult,
    ValidationStatus, ProcessingStatus, RePromptStrategy, RePromptTemplate
)
from .constraints import ConstraintEngine, ConstraintViolationException
from .validator import OutputValidator
from .processor import OutputProcessor


class OutputController:
    """输出控制器主类"""
    
    def __init__(
        self,
        config: OutputControlConfig,
        llm_client=None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger or self._setup_logger()
        
        # 初始化各个组件
        self.constraint_engine = ConstraintEngine(config.pre_constraints, self.logger)
        self.validator = OutputValidator(config.validation_rules, llm_client, self.logger)
        self.processor = OutputProcessor(config.processing_rules, self.logger)
        
        # Re-prompt相关
        self.re_prompt_templates = config.re_prompt_templates
        
        self.logger.info("输出控制器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers and self.config.logging_enabled:
            logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
            
            # 创建日志文件处理器
            if self.config.log_file:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setLevel(logger.level)
                
                # 创建格式化器
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger.level)
            console_handler.setFormatter(
                logging.Formatter('%(levelname)s - %(message)s')
            )
            logger.addHandler(console_handler)
        
        return logger
    
    async def control_output(
        self,
        original_prompt: str,
        agent_output: str,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = None
    ) -> ControlResult:
        """
        控制Agent输出的主要方法
        
        Args:
            original_prompt: 原始提示
            agent_output: Agent的原始输出
            context: 上下文信息
            output_type: 输出类型
            
        Returns:
            ControlResult: 控制结果
        """
        if not self.config.enabled:
            return ControlResult(
                success=True,
                final_output=agent_output,
                original_output=agent_output
            )
        
        start_time = time.time()
        context = context or {}
        context['output_type'] = output_type
        context['original_prompt'] = original_prompt
        
        self.logger.info(f"开始输出控制流程，输出类型: {output_type}")
        
        result = ControlResult(
            success=False,
            final_output=agent_output,
            original_output=agent_output,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # 阶段1: 前置约束检查
            constraint_results = await self._check_constraints(agent_output, context)
            result.constraint_violations = constraint_results
            
            # 检查是否有必需约束失败
            critical_failures = [
                r for r in constraint_results 
                if r.status == ValidationStatus.FAILED and self._is_critical_constraint(r.rule_name)
            ]
            
            current_output = agent_output
            
            # 如果有关键约束失败，尝试re-prompt
            if critical_failures and self.config.re_prompt_enabled:
                self.logger.warning(f"发现 {len(critical_failures)} 个关键约束失败，尝试re-prompt")
                
                re_prompt_result = await self._handle_constraint_failures(
                    original_prompt, current_output, critical_failures, context
                )
                
                if re_prompt_result['success']:
                    current_output = re_prompt_result['output']
                    result.re_prompt_attempts = re_prompt_result['attempts']
                    result.re_prompt_history = re_prompt_result['history']
                    
                    # 重新检查约束
                    constraint_results = await self._check_constraints(current_output, context)
                    result.constraint_violations = constraint_results
            
            # 阶段2: 验证
            validation_results = await self._validate_output(current_output, context)
            result.validation_results = validation_results
            
            # 检查验证失败
            validation_failures = [
                r for r in validation_results 
                if r.status == ValidationStatus.FAILED
            ]
            
            # 如果有验证失败，尝试re-prompt
            if validation_failures and self.config.re_prompt_enabled and result.re_prompt_attempts < self.config.re_prompt_max_attempts:
                self.logger.warning(f"发现 {len(validation_failures)} 个验证失败，尝试re-prompt")
                
                re_prompt_result = await self._handle_validation_failures(
                    original_prompt, current_output, validation_failures, context
                )
                
                if re_prompt_result['success']:
                    current_output = re_prompt_result['output']
                    result.re_prompt_attempts += re_prompt_result['attempts']
                    result.re_prompt_history.extend(re_prompt_result['history'])
                    
                    # 重新验证
                    validation_results = await self._validate_output(current_output, context)
                    result.validation_results = validation_results
            
            # 阶段3: 后处理
            processed_output, processing_results = await self._process_output(current_output, context)
            result.processing_results = processing_results
            result.final_output = processed_output
            
            # 判断整体成功性
            has_critical_failures = any(
                r.status == ValidationStatus.FAILED and self._is_critical_constraint(r.rule_name)
                for r in result.constraint_violations
            )
            has_validation_failures = any(
                r.status == ValidationStatus.FAILED 
                for r in result.validation_results
            )
            has_processing_failures = any(
                r.status == ProcessingStatus.FAILED 
                for r in result.processing_results
            )
            
            result.success = not (has_critical_failures or has_validation_failures or has_processing_failures)
            
        except Exception as e:
            self.logger.error(f"输出控制流程异常: {e}")
            result.success = False
            result.debug_info['error'] = str(e)
        
        finally:
            result.processing_time = time.time() - start_time
            
            if self.config.debug_mode:
                result.debug_info.update({
                    'config': self.config.__dict__,
                    'context': context
                })
        
        self.logger.info(f"输出控制完成: 成功={result.success}, 用时={result.processing_time:.2f}s")
        return result
    
    async def _check_constraints(self, output: str, context: Dict[str, Any]) -> List[ValidationResult]:
        """检查前置约束"""
        try:
            return self.constraint_engine.check_constraints(output, context)
        except Exception as e:
            self.logger.error(f"约束检查异常: {e}")
            return []
    
    async def _validate_output(self, output: str, context: Dict[str, Any]) -> List[ValidationResult]:
        """验证输出"""
        try:
            return await self.validator.validate_output(output, context)
        except Exception as e:
            self.logger.error(f"输出验证异常: {e}")
            return []
    
    async def _process_output(self, output: str, context: Dict[str, Any]) -> Tuple[str, List[ProcessingResult]]:
        """处理输出"""
        try:
            return await self.processor.process_output(output, context)
        except Exception as e:
            self.logger.error(f"输出处理异常: {e}")
            return output, []
    
    def _is_critical_constraint(self, constraint_name: str) -> bool:
        """判断是否是关键约束"""
        # 查找对应的约束配置
        for constraint in self.config.pre_constraints:
            if constraint.name == constraint_name:
                return constraint.required
        return False
    
    async def _handle_constraint_failures(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理约束失败"""
        attempts = 0
        history = []
        
        for attempt in range(self.config.re_prompt_max_attempts):
            attempts += 1
            
            # 构建re-prompt
            re_prompt = self._build_constraint_re_prompt(
                original_prompt, current_output, failures, context, attempt
            )
            
            # 记录历史
            history.append({
                'attempt': attempt + 1,
                'type': 'constraint_failure',
                'failures': [f.rule_name for f in failures],
                'prompt': re_prompt
            })
            
            try:
                # 调用LLM重新生成
                new_output = await self._call_llm_for_re_prompt(re_prompt)
                
                # 检查新输出的约束
                new_constraint_results = await self._check_constraints(new_output, context)
                new_failures = [
                    r for r in new_constraint_results 
                    if r.status == ValidationStatus.FAILED and self._is_critical_constraint(r.rule_name)
                ]
                
                if not new_failures:
                    # 成功修复
                    history[-1]['result'] = 'success'
                    history[-1]['output_length'] = len(new_output)
                    
                    return {
                        'success': True,
                        'output': new_output,
                        'attempts': attempts,
                        'history': history
                    }
                else:
                    # 仍有失败
                    history[-1]['result'] = 'partial'
                    history[-1]['remaining_failures'] = [f.rule_name for f in new_failures]
                    current_output = new_output
                    failures = new_failures
                    
            except Exception as e:
                self.logger.error(f"Re-prompt异常 (尝试 {attempt + 1}): {e}")
                history[-1]['result'] = 'error'
                history[-1]['error'] = str(e)
        
        return {
            'success': False,
            'output': current_output,
            'attempts': attempts,
            'history': history
        }
    
    async def _handle_validation_failures(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理验证失败"""
        attempts = 0
        history = []
        
        for attempt in range(self.config.re_prompt_max_attempts):
            attempts += 1
            
            # 构建re-prompt
            re_prompt = self._build_validation_re_prompt(
                original_prompt, current_output, failures, context, attempt
            )
            
            # 记录历史
            history.append({
                'attempt': attempt + 1,
                'type': 'validation_failure',
                'failures': [f.rule_name for f in failures],
                'prompt': re_prompt
            })
            
            try:
                # 调用LLM重新生成
                new_output = await self._call_llm_for_re_prompt(re_prompt)
                
                # 检查新输出的验证
                new_validation_results = await self._validate_output(new_output, context)
                new_failures = [
                    r for r in new_validation_results 
                    if r.status == ValidationStatus.FAILED
                ]
                
                if not new_failures:
                    # 成功修复
                    history[-1]['result'] = 'success'
                    history[-1]['output_length'] = len(new_output)
                    
                    return {
                        'success': True,
                        'output': new_output,
                        'attempts': attempts,
                        'history': history
                    }
                else:
                    # 仍有失败
                    history[-1]['result'] = 'partial'
                    history[-1]['remaining_failures'] = [f.rule_name for f in new_failures]
                    current_output = new_output
                    failures = new_failures
                    
            except Exception as e:
                self.logger.error(f"验证Re-prompt异常 (尝试 {attempt + 1}): {e}")
                history[-1]['result'] = 'error'
                history[-1]['error'] = str(e)
        
        return {
            'success': False,
            'output': current_output,
            'attempts': attempts,
            'history': history
        }
    
    def _build_constraint_re_prompt(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult],
        context: Dict[str, Any],
        attempt: int
    ) -> str:
        """构建约束失败的re-prompt"""
        template_name = 'format_fix'  # 默认使用格式修复模板
        
        # 根据失败类型选择模板
        if any('content' in f.rule_name.lower() for f in failures):
            template_name = 'content_fix'
        
        template = self.re_prompt_templates.get(template_name)
        if not template:
            # 使用默认模板
            return self._build_default_constraint_re_prompt(original_prompt, current_output, failures)
        
        # 构建错误信息
        validation_errors = []
        format_requirements = []
        
        for failure in failures:
            validation_errors.append(f"- {failure.rule_name}: {failure.message}")
            if failure.suggestions:
                validation_errors.extend([f"  建议: {s}" for s in failure.suggestions])
        
        # 获取格式要求
        format_requirements.append(self.constraint_engine.get_constraint_prompt_additions())
        
        return template.template.format(
            original_question=original_prompt,
            format_requirements='\n'.join(format_requirements),
            validation_errors='\n'.join(validation_errors)
        )
    
    def _build_validation_re_prompt(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult],
        context: Dict[str, Any],
        attempt: int
    ) -> str:
        """构建验证失败的re-prompt"""
        template_name = 'quality_improvement'
        
        template = self.re_prompt_templates.get(template_name)
        if not template:
            return self._build_default_validation_re_prompt(original_prompt, current_output, failures)
        
        # 构建质量评估信息
        quality_assessment = []
        improvement_suggestions = []
        
        for failure in failures:
            quality_assessment.append(f"- {failure.rule_name}: {failure.message}")
            if failure.score is not None:
                quality_assessment.append(f"  评分: {failure.score}")
            if failure.suggestions:
                improvement_suggestions.extend(failure.suggestions)
        
        return template.template.format(
            original_question=original_prompt,
            previous_answer=current_output,
            quality_assessment='\n'.join(quality_assessment),
            improvement_suggestions='\n'.join(improvement_suggestions)
        )
    
    def _build_default_constraint_re_prompt(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult]
    ) -> str:
        """构建默认约束re-prompt"""
        error_messages = [f"- {f.rule_name}: {f.message}" for f in failures]
        
        return f"""你的上一个回答不符合要求，存在以下问题：

{chr(10).join(error_messages)}

原始问题：{original_prompt}

请根据上述问题重新生成回答，确保符合所有约束要求。"""
    
    def _build_default_validation_re_prompt(
        self,
        original_prompt: str,
        current_output: str,
        failures: List[ValidationResult]
    ) -> str:
        """构建默认验证re-prompt"""
        error_messages = [f"- {f.rule_name}: {f.message}" for f in failures]
        
        return f"""你的回答需要改进，存在以下问题：

{chr(10).join(error_messages)}

原始问题：{original_prompt}
你的回答：{current_output}

请基于上述反馈重新优化你的回答。"""
    
    async def _call_llm_for_re_prompt(self, prompt: str) -> str:
        """调用LLM进行re-prompt"""
        if not self.llm_client:
            raise ValueError("LLM客户端未配置，无法执行re-prompt")
        
        try:
            if hasattr(self.llm_client, 'chat_completions_create'):
                response = await self.llm_client.chat_completions_create(
                    model="gpt-4o-mini",  # 使用快速模型
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3  # 较低温度以保持一致性
                )
                return response.choices[0].message.content
            elif hasattr(self.llm_client, 'generate'):
                response = await self.llm_client.generate(prompt)
                return response
            else:
                raise ValueError("不支持的LLM客户端")
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise
    
    def get_constraint_prompt_additions(self, output_type: Optional[str] = None) -> str:
        """获取约束条件的Prompt补充"""
        return self.constraint_engine.get_constraint_prompt_additions(output_type)
    
    def get_control_summary(self, result: ControlResult) -> Dict[str, Any]:
        """获取控制摘要"""
        constraint_summary = {
            'total': len(result.constraint_violations),
            'failed': sum(1 for r in result.constraint_violations if r.status == ValidationStatus.FAILED),
            'warnings': sum(1 for r in result.constraint_violations if r.status == ValidationStatus.WARNING)
        }
        
        validation_summary = self.validator.get_validation_summary(result.validation_results)
        processing_summary = self.processor.get_processing_summary(result.processing_results)
        
        return {
            'success': result.success,
            'processing_time': result.processing_time,
            're_prompt_attempts': result.re_prompt_attempts,
            'constraints': constraint_summary,
            'validation': validation_summary,
            'processing': processing_summary,
            'output_length': len(result.final_output),
            'improvements_made': len(processing_summary.get('changes_made', []))
        }

    @classmethod
    def from_config_file(cls, config_path: str, llm_client=None, logger=None) -> 'OutputController':
        """从配置文件创建控制器"""
        config = OutputControlConfig.from_yaml(config_path)
        return cls(config, llm_client, logger) 