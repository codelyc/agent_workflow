"""
后处理系统实现

对Agent输出进行后处理，包括文本清理、格式化、内容补全、敏感信息过滤等
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .models import (
    ProcessingRule, ProcessingResult, ProcessingStatus
)


class OutputProcessor:
    """输出处理器"""
    
    def __init__(self, processing_rules: List[ProcessingRule], logger: Optional[logging.Logger] = None):
        self.processing_rules = processing_rules
        self.logger = logger or logging.getLogger(__name__)
        
        # 按类型分组处理规则
        self.text_rules = [r for r in processing_rules if r.type == 'text' and r.enabled]
        self.format_rules = [r for r in processing_rules if r.type == 'format' and r.enabled]
        self.completion_rules = [r for r in processing_rules if r.type == 'completion' and r.enabled]
        self.filter_rules = [r for r in processing_rules if r.type == 'filter' and r.enabled]
    
    async def process_output(self, output: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ProcessingResult]]:
        """处理输出"""
        processed_output = output
        results = []
        context = context or {}
        
        self.logger.info(f"开始后处理，原始长度: {len(output)}")
        
        # 文本处理
        for rule in self.text_rules:
            processed_output, result = await self._process_text(processed_output, rule, context)
            results.append(result)
        
        # 格式化处理
        for rule in self.format_rules:
            processed_output, result = await self._process_format(processed_output, rule, context)
            results.append(result)
        
        # 补全处理
        for rule in self.completion_rules:
            processed_output, result = await self._process_completion(processed_output, rule, context)
            results.append(result)
        
        # 过滤处理
        for rule in self.filter_rules:
            processed_output, result = await self._process_filter(processed_output, rule, context)
            results.append(result)
        
        # 统计结果
        successful_count = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        failed_count = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        
        self.logger.info(f"后处理完成: 成功 {successful_count}, 失败 {failed_count}, 最终长度: {len(processed_output)}")
        
        return processed_output, results
    
    async def _process_text(self, output: str, rule: ProcessingRule, context: Dict[str, Any]) -> Tuple[str, ProcessingResult]:
        """文本处理"""
        try:
            original_output = output
            changes_made = []
            
            # 去除多余空白
            if rule.strip_whitespace:
                output = output.strip()
                if output != original_output:
                    changes_made.append("去除首尾空白")
            
            # 标准化行结束符
            if rule.normalize_line_endings:
                original_lines = len(output.split('\n'))
                output = output.replace('\r\n', '\n').replace('\r', '\n')
                new_lines = len(output.split('\n'))
                if original_lines != new_lines:
                    changes_made.append("标准化行结束符")
            
            # 移除空行
            if rule.remove_empty_lines:
                lines = output.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                if len(non_empty_lines) != len(lines):
                    output = '\n'.join(non_empty_lines)
                    changes_made.append(f"移除 {len(lines) - len(non_empty_lines)} 个空行")
            
            return output, ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                rule_name=rule.name,
                processed_content=output,
                original_content=original_output,
                changes_made=changes_made,
                message=f"文本处理完成: {'; '.join(changes_made)}" if changes_made else "文本无需处理"
            )
            
        except Exception as e:
            self.logger.error(f"文本处理异常 {rule.name}: {e}")
            return output, ProcessingResult(
                status=ProcessingStatus.FAILED,
                rule_name=rule.name,
                processed_content=output,
                original_content=output,
                message=f"文本处理异常: {str(e)}"
            )
    
    async def _process_format(self, output: str, rule: ProcessingRule, context: Dict[str, Any]) -> Tuple[str, ProcessingResult]:
        """格式化处理"""
        try:
            original_output = output
            changes_made = []
            
            # JSON格式化
            if rule.auto_format_json:
                try:
                    # 检查是否是JSON
                    parsed = json.loads(output.strip())
                    formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
                    if formatted_json != output.strip():
                        output = formatted_json
                        changes_made.append("JSON格式化")
                except json.JSONDecodeError:
                    # 不是JSON，跳过
                    pass
            
            # Markdown格式化
            if rule.auto_format_markdown:
                formatted_md = self._format_markdown(output)
                if formatted_md != output:
                    output = formatted_md
                    changes_made.append("Markdown格式化")
            
            # 代码块美化
            if rule.prettify_code_blocks:
                formatted_code = self._prettify_code_blocks(output)
                if formatted_code != output:
                    output = formatted_code
                    changes_made.append("代码块美化")
            
            return output, ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                rule_name=rule.name,
                processed_content=output,
                original_content=original_output,
                changes_made=changes_made,
                message=f"格式化处理完成: {'; '.join(changes_made)}" if changes_made else "格式无需调整"
            )
            
        except Exception as e:
            self.logger.error(f"格式化处理异常 {rule.name}: {e}")
            return output, ProcessingResult(
                status=ProcessingStatus.FAILED,
                rule_name=rule.name,
                processed_content=output,
                original_content=output,
                message=f"格式化处理异常: {str(e)}"
            )
    
    def _format_markdown(self, content: str) -> str:
        """格式化Markdown"""
        lines = content.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            # 确保标题前后有空行
            if line.startswith('#'):
                if i > 0 and lines[i-1].strip() != '':
                    formatted_lines.append('')
                formatted_lines.append(line)
                if i < len(lines) - 1 and lines[i+1].strip() != '':
                    formatted_lines.append('')
            # 确保列表项格式正确
            elif re.match(r'^\s*[\*\-\+]\s+', line):
                formatted_lines.append(line)
            # 确保数字列表格式正确
            elif re.match(r'^\s*\d+\.\s+', line):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _prettify_code_blocks(self, content: str) -> str:
        """美化代码块"""
        # 查找代码块
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        
        def format_code_block(match):
            language = match.group(1) or ''
            code = match.group(2)
            
            # 简单的代码格式化
            if language.lower() in ['python', 'py']:
                # Python代码格式化（简单实现）
                formatted_code = self._format_python_code(code)
            elif language.lower() in ['json']:
                # JSON格式化
                try:
                    parsed = json.loads(code)
                    formatted_code = json.dumps(parsed, indent=2, ensure_ascii=False)
                except:
                    formatted_code = code
            else:
                formatted_code = code
            
            return f'```{language}\n{formatted_code}\n```'
        
        return re.sub(code_block_pattern, format_code_block, content, flags=re.DOTALL)
    
    def _format_python_code(self, code: str) -> str:
        """简单的Python代码格式化"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # 处理缩进
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                formatted_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            elif stripped.startswith(('except', 'elif', 'else:')):
                formatted_lines.append('    ' * (indent_level - 1) + stripped)
            elif stripped in ['pass', 'break', 'continue'] or stripped.startswith('return'):
                formatted_lines.append('    ' * indent_level + stripped)
                if stripped.startswith('return'):
                    indent_level = max(0, indent_level - 1)
            else:
                formatted_lines.append('    ' * indent_level + stripped)
        
        return '\n'.join(formatted_lines)
    
    async def _process_completion(self, output: str, rule: ProcessingRule, context: Dict[str, Any]) -> Tuple[str, ProcessingResult]:
        """补全处理"""
        try:
            original_output = output
            changes_made = []
            
            # 添加缺失的章节
            if rule.add_missing_sections:
                added_sections = self._add_missing_sections(output, context)
                if added_sections:
                    output = added_sections
                    changes_made.append("添加缺失章节")
            
            # 生成摘要
            if rule.generate_summary:
                summary = self._generate_summary(output)
                if summary and "摘要" not in output:
                    output = f"## 摘要\n\n{summary}\n\n{output}"
                    changes_made.append("生成摘要")
            
            # 添加元数据
            if rule.add_metadata:
                metadata = self._generate_metadata(output, context)
                if metadata:
                    output = f"{metadata}\n\n{output}"
                    changes_made.append("添加元数据")
            
            return output, ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                rule_name=rule.name,
                processed_content=output,
                original_content=original_output,
                changes_made=changes_made,
                message=f"补全处理完成: {'; '.join(changes_made)}" if changes_made else "无需补全"
            )
            
        except Exception as e:
            self.logger.error(f"补全处理异常 {rule.name}: {e}")
            return output, ProcessingResult(
                status=ProcessingStatus.FAILED,
                rule_name=rule.name,
                processed_content=output,
                original_content=output,
                message=f"补全处理异常: {str(e)}"
            )
    
    def _add_missing_sections(self, output: str, context: Dict[str, Any]) -> Optional[str]:
        """添加缺失的章节"""
        # 检查常见的分析报告章节
        required_sections = ['背景', '分析', '结论', '建议']
        existing_sections = []
        
        for section in required_sections:
            if section in output or section.lower() in output.lower():
                existing_sections.append(section)
        
        missing_sections = [s for s in required_sections if s not in existing_sections]
        
        if missing_sections:
            # 添加缺失章节的占位符
            additional_content = []
            for section in missing_sections:
                additional_content.append(f"\n## {section}\n\n[待补充内容]\n")
            
            return output + '\n'.join(additional_content)
        
        return None
    
    def _generate_summary(self, output: str) -> Optional[str]:
        """生成摘要（简单实现）"""
        # 提取第一段作为摘要
        paragraphs = [p.strip() for p in output.split('\n\n') if p.strip()]
        if paragraphs:
            first_paragraph = paragraphs[0]
            # 限制摘要长度
            if len(first_paragraph) > 200:
                return first_paragraph[:200] + "..."
            return first_paragraph
        return None
    
    def _generate_metadata(self, output: str, context: Dict[str, Any]) -> Optional[str]:
        """生成元数据"""
        metadata_parts = []
        
        # 添加时间戳
        from datetime import datetime
        metadata_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 添加字数统计
        word_count = len(output.split())
        char_count = len(output)
        metadata_parts.append(f"字数统计: {word_count} 词, {char_count} 字符")
        
        # 添加上下文信息
        if context.get('output_type'):
            metadata_parts.append(f"输出类型: {context['output_type']}")
        
        if metadata_parts:
            return "---\n" + "\n".join(metadata_parts) + "\n---"
        
        return None
    
    async def _process_filter(self, output: str, rule: ProcessingRule, context: Dict[str, Any]) -> Tuple[str, ProcessingResult]:
        """过滤处理"""
        try:
            original_output = output
            changes_made = []
            
            # 移除敏感信息
            if rule.remove_sensitive_info:
                filtered_output, sensitive_items = self._remove_sensitive_info(output)
                if sensitive_items:
                    output = filtered_output
                    changes_made.append(f"移除敏感信息: {', '.join(sensitive_items)}")
            
            # 过滤不当内容
            if rule.filter_inappropriate_content:
                filtered_output, inappropriate_items = self._filter_inappropriate_content(output)
                if inappropriate_items:
                    output = filtered_output
                    changes_made.append(f"过滤不当内容: {', '.join(inappropriate_items)}")
            
            return output, ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                rule_name=rule.name,
                processed_content=output,
                original_content=original_output,
                changes_made=changes_made,
                message=f"过滤处理完成: {'; '.join(changes_made)}" if changes_made else "无需过滤"
            )
            
        except Exception as e:
            self.logger.error(f"过滤处理异常 {rule.name}: {e}")
            return output, ProcessingResult(
                status=ProcessingStatus.FAILED,
                rule_name=rule.name,
                processed_content=output,
                original_content=output,
                message=f"过滤处理异常: {str(e)}"
            )
    
    def _remove_sensitive_info(self, output: str) -> Tuple[str, List[str]]:
        """移除敏感信息"""
        sensitive_patterns = {
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[邮箱地址]'),
            'phone': (r'\b(?:\+86|86)?[-\s]?1[3-9]\d{9}\b', '[电话号码]'),
            'id_card': (r'\b\d{17}[\dX]\b', '[身份证号]'),
            'api_key': (r'\b[Aa][Pp][Ii][-_]?[Kk][Ee][Yy][-_:=]\s*["\']?[\w\-]{16,}["\']?', '[API密钥]'),
            'password': (r'\b[Pp]assword[-_:=]\s*["\']?[\w\-]{6,}["\']?', '[密码]'),
        }
        
        removed_items = []
        filtered_output = output
        
        for item_type, (pattern, replacement) in sensitive_patterns.items():
            matches = re.findall(pattern, filtered_output)
            if matches:
                filtered_output = re.sub(pattern, replacement, filtered_output)
                removed_items.append(f"{item_type}({len(matches)}个)")
        
        return filtered_output, removed_items
    
    def _filter_inappropriate_content(self, output: str) -> Tuple[str, List[str]]:
        """过滤不当内容"""
        inappropriate_patterns = [
            r'\b[不当词汇1]\b',
            r'\b[不当词汇2]\b',
            # 添加更多不当内容模式
        ]
        
        filtered_items = []
        filtered_output = output
        
        for pattern in inappropriate_patterns:
            matches = re.findall(pattern, filtered_output, re.IGNORECASE)
            if matches:
                filtered_output = re.sub(pattern, '[已过滤]', filtered_output, flags=re.IGNORECASE)
                filtered_items.extend(matches)
        
        return filtered_output, filtered_items
    
    def get_processing_summary(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """获取处理摘要"""
        total_count = len(results)
        success_count = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        failed_count = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        partial_count = sum(1 for r in results if r.status == ProcessingStatus.PARTIAL)
        
        # 收集所有变更
        all_changes = []
        for result in results:
            all_changes.extend(result.changes_made)
        
        return {
            'total_count': total_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'partial_count': partial_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'has_failures': failed_count > 0,
            'changes_made': all_changes,
            'failed_rules': [r.rule_name for r in results if r.status == ProcessingStatus.FAILED]
        }


class ProcessingPipeline:
    """处理管道"""
    
    def __init__(self, processors: List[OutputProcessor]):
        self.processors = processors
        self.logger = logging.getLogger(__name__)
    
    async def run_processing(self, output: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[ProcessingResult]]:
        """运行处理管道"""
        processed_output = output
        all_results = []
        
        for processor in self.processors:
            try:
                processed_output, results = await processor.process_output(processed_output, context)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"处理器运行失败: {e}")
                # 继续运行其他处理器
        
        return processed_output, all_results 