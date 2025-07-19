"""
代码助手AgentWorker

专门用于代码生成、代码优化、代码解释、代码审查等任务的Agent工作流
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

# 添加项目根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from core.output_control import OutputController
from core.config.llm_manager import LLMManager


class CodeAssistantWorker:
    """代码助手工作器"""
    
    def __init__(self, llm_model: str = "gpt-4o"):
        """
        初始化代码助手工作器
        
        Args:
            llm_model: 使用的LLM模型，代码生成推荐使用强模型
        """
        self.llm_model = llm_model
        self.llm_manager = LLMManager()
        
        # 从本地配置加载输出控制器
        config_path = Path(__file__).parent / "configs" / "output_control.yaml"
        self.output_controller = OutputController.from_config_file(
            str(config_path),
            llm_client=self.llm_manager.get_client()
        )
        
        print(f"✅ 代码助手工作器初始化完成，使用模型: {llm_model}")
    
    async def generate_function(self, function_description: str, language: str = "python", 
                               include_tests: bool = True) -> Dict[str, Any]:
        """
        生成函数代码
        
        Args:
            function_description: 函数功能描述
            language: 编程语言
            include_tests: 是否包含测试代码
            
        Returns:
            包含生成的函数代码和质量信息的字典
        """
        test_requirement = "并提供相应的单元测试代码" if include_tests else ""
        
        prompt = f"""请基于以下描述生成{language}函数{test_requirement}：

功能描述：{function_description}
编程语言：{language}

要求：
1. 函数定义清晰，参数和返回值有类型注解（如果语言支持）
2. 包含详细的文档字符串说明函数用途、参数、返回值
3. 代码逻辑清晰，遵循最佳实践
4. 包含适当的错误处理
5. 使用代码块格式，包含语法高亮
{f'6. 提供完整的单元测试用例' if include_tests else ''}

请确保代码安全、高效，避免使用危险函数。"""

        # 调用LLM生成代码
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="function"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "function_description": function_description,
            "language": language,
            "include_tests": include_tests,
            "quality_score": self._calculate_code_quality(control_result),
            "security_score": self._assess_security(control_result),
            "processing_time": control_result.processing_time,
            "processing_info": self._get_processing_info(control_result)
        }
    
    async def generate_class(self, class_description: str, language: str = "python",
                            include_examples: bool = True) -> Dict[str, Any]:
        """
        生成类代码
        
        Args:
            class_description: 类功能描述
            language: 编程语言
            include_examples: 是否包含使用示例
            
        Returns:
            包含生成的类代码和质量信息的字典
        """
        example_requirement = "并提供使用示例" if include_examples else ""
        
        prompt = f"""请基于以下描述生成{language}类{example_requirement}：

类描述：{class_description}
编程语言：{language}

要求：
1. 类设计合理，遵循单一职责原则
2. 包含必要的构造函数、属性和方法
3. 每个方法都有清晰的文档说明
4. 使用适当的访问修饰符（如果语言支持）
5. 包含错误处理和参数验证
6. 遵循命名规范和代码风格
{f'7. 提供详细的使用示例代码' if include_examples else ''}

请确保代码结构清晰，易于维护和扩展。"""

        # 调用LLM生成代码
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="class"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "class_description": class_description,
            "language": language,
            "include_examples": include_examples,
            "quality_score": self._calculate_code_quality(control_result),
            "design_quality": self._assess_design_quality(control_result)
        }
    
    async def optimize_code(self, code: str, optimization_goals: List[str]) -> Dict[str, Any]:
        """
        优化代码
        
        Args:
            code: 待优化的代码
            optimization_goals: 优化目标列表（性能、可读性、安全性等）
            
        Returns:
            包含优化后代码和改进说明的字典
        """
        goals_text = "、".join(optimization_goals)
        
        prompt = f"""请对以下代码进行优化，优化重点：{goals_text}

原始代码：
```
{code}
```

优化要求：
1. 保持原有功能不变
2. 针对指定目标进行优化：{goals_text}
3. 提供优化后的完整代码
4. 详细说明每个优化点和改进理由
5. 指出潜在的问题和风险
6. 给出性能或安全性方面的建议

请确保优化后的代码更加高效、安全、可维护。"""

        # 调用LLM进行优化
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="script"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "original_code": code,
            "optimization_goals": optimization_goals,
            "quality_score": self._calculate_code_quality(control_result),
            "optimization_effectiveness": self._assess_optimization(control_result, code)
        }
    
    async def explain_code(self, code: str, detail_level: str = "intermediate") -> Dict[str, Any]:
        """
        解释代码
        
        Args:
            code: 需要解释的代码
            detail_level: 详细程度 (beginner/intermediate/advanced)
            
        Returns:
            包含代码解释的字典
        """
        level_guide = {
            "beginner": "面向初学者，解释基础概念和语法",
            "intermediate": "面向有一定基础的开发者，重点说明逻辑和思路",
            "advanced": "面向高级开发者，深入分析算法、设计模式和优化点"
        }
        
        prompt = f"""请解释以下代码，解释详细程度：{level_guide.get(detail_level, '中等')}

代码：
```
{code}
```

解释要求：
1. 概述代码的主要功能和用途
2. 逐步解释关键代码段的作用
3. 说明算法逻辑和数据流程
4. 指出代码中的重要技术点
5. 分析代码的优缺点
6. 提供改进建议（如有）

请用通俗易懂的语言进行解释，帮助理解代码的工作原理。"""

        # 调用LLM进行解释
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制（使用通用report类型）
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="script"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "original_code": code,
            "detail_level": detail_level,
            "explanation_quality": self._assess_explanation_quality(control_result)
        }
    
    async def review_code(self, code: str, review_focus: List[str] = None) -> Dict[str, Any]:
        """
        代码审查
        
        Args:
            code: 需要审查的代码
            review_focus: 审查重点列表
            
        Returns:
            包含审查结果的字典
        """
        if review_focus is None:
            review_focus = ["代码质量", "安全性", "性能", "可维护性"]
        
        focus_text = "、".join(review_focus)
        
        prompt = f"""请对以下代码进行专业审查，重点关注：{focus_text}

代码：
```
{code}
```

审查要求：
1. 代码质量评估：语法、结构、命名规范
2. 安全性检查：潜在安全漏洞和风险点
3. 性能分析：性能瓶颈和优化机会
4. 可维护性评估：代码的可读性和可扩展性
5. 最佳实践检查：是否遵循行业标准和规范
6. 具体改进建议：提供可操作的优化方案

请给出专业、客观的审查意见和改进建议。"""

        # 调用LLM进行审查
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="script"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "original_code": code,
            "review_focus": review_focus,
            "review_quality": self._assess_review_quality(control_result),
            "security_issues": self._extract_security_issues(control_result)
        }
    
    async def generate_api(self, api_description: str, framework: str = "FastAPI",
                          include_docs: bool = True) -> Dict[str, Any]:
        """
        生成API接口代码
        
        Args:
            api_description: API功能描述
            framework: 使用的框架
            include_docs: 是否包含API文档
            
        Returns:
            包含API代码的字典
        """
        docs_requirement = "包含完整的API文档和使用示例" if include_docs else ""
        
        prompt = f"""请基于以下描述生成{framework} API接口代码：

API描述：{api_description}
使用框架：{framework}

要求：
1. 完整的API路由定义
2. 请求和响应模型定义
3. 适当的HTTP状态码处理
4. 输入验证和错误处理
5. 安全性考虑（认证、授权等）
6. 代码注释和类型注解
{f'7. {docs_requirement}' if include_docs else ''}

请确保API设计符合RESTful规范，代码安全可靠。"""

        # 调用LLM生成API代码
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="api"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "api_description": api_description,
            "framework": framework,
            "include_docs": include_docs,
            "quality_score": self._calculate_code_quality(control_result),
            "api_design_score": self._assess_api_design(control_result)
        }
    
    def _calculate_code_quality(self, control_result) -> float:
        """计算代码质量分数"""
        if not control_result.success:
            return 0.0
        
        # 基础分数
        base_score = 8.0
        
        # 检查安全性约束
        security_violations = [r for r in control_result.constraint_violations 
                             if 'security' in r.rule_name.lower() and r.status.value == 'failed']
        if security_violations:
            base_score -= 2.0  # 安全问题严重扣分
        
        # 检查AI代码审查结果
        ai_results = [r for r in control_result.validation_results 
                     if r.rule_name == 'ai_self_check' and r.details]
        if ai_results:
            ai_details = ai_results[0].details.get('ai_check_results', [])
            for check in ai_details:
                if check.get('type') == 'code_quality':
                    syntax_score = check.get('syntax', 0)
                    security_score = check.get('security', 0)
                    readability_score = check.get('readability', 0)
                    # 加权平均
                    weighted_score = (syntax_score * 0.3 + security_score * 0.4 + readability_score * 0.3) / 10 * 2
                    base_score = max(base_score, base_score + weighted_score - 1.0)
        
        return max(0.0, min(10.0, base_score))
    
    def _assess_security(self, control_result) -> float:
        """评估代码安全性"""
        if not control_result.success:
            return 0.0
        
        # 检查安全相关的约束违反
        security_violations = [r for r in control_result.constraint_violations 
                             if 'security' in r.rule_name.lower() and r.status.value == 'failed']
        
        # 基础安全分数
        base_score = 10.0
        
        # 每个安全违反扣分
        base_score -= len(security_violations) * 2.0
        
        # 检查AI安全检查结果
        ai_results = [r for r in control_result.validation_results 
                     if r.rule_name == 'ai_self_check' and r.details]
        if ai_results:
            ai_details = ai_results[0].details.get('ai_check_results', [])
            for check in ai_details:
                if check.get('type') == 'security_check' and not check.get('secure', True):
                    base_score -= 1.0
        
        return max(0.0, min(10.0, base_score))
    
    def _assess_design_quality(self, control_result) -> str:
        """评估设计质量"""
        quality_score = self._calculate_code_quality(control_result)
        if quality_score >= 8.5:
            return "优秀"
        elif quality_score >= 7.0:
            return "良好"
        elif quality_score >= 6.0:
            return "一般"
        else:
            return "需改进"
    
    def _assess_optimization(self, control_result, original_code: str) -> str:
        """评估优化效果"""
        if not control_result.success:
            return "失败"
        
        # 简单的优化效果评估
        final_code = control_result.final_output
        
        # 检查是否包含优化说明
        if "优化" in final_code and "改进" in final_code:
            return "显著"
        elif "改进" in final_code or "优化" in final_code:
            return "中等"
        else:
            return "轻微"
    
    def _assess_explanation_quality(self, control_result) -> str:
        """评估解释质量"""
        if not control_result.success:
            return "差"
        
        content = control_result.final_output.lower()
        quality_indicators = ["功能", "逻辑", "算法", "原理", "作用"]
        found_indicators = sum(1 for indicator in quality_indicators if indicator in content)
        
        ratio = found_indicators / len(quality_indicators)
        if ratio >= 0.8:
            return "优秀"
        elif ratio >= 0.6:
            return "良好"
        elif ratio >= 0.4:
            return "一般"
        else:
            return "差"
    
    def _assess_review_quality(self, control_result) -> str:
        """评估审查质量"""
        if not control_result.success:
            return "差"
        
        content = control_result.final_output.lower()
        review_aspects = ["质量", "安全", "性能", "维护", "建议"]
        found_aspects = sum(1 for aspect in review_aspects if aspect in content)
        
        ratio = found_aspects / len(review_aspects)
        if ratio >= 0.8:
            return "全面"
        elif ratio >= 0.6:
            return "较好"
        else:
            return "简单"
    
    def _extract_security_issues(self, control_result) -> List[str]:
        """提取安全问题"""
        issues = []
        
        # 从约束违反中提取安全问题
        security_violations = [r for r in control_result.constraint_violations 
                             if 'security' in r.rule_name.lower() and r.status.value == 'failed']
        for violation in security_violations:
            issues.append(violation.message)
        
        return issues
    
    def _assess_api_design(self, control_result) -> str:
        """评估API设计质量"""
        if not control_result.success:
            return "差"
        
        content = control_result.final_output.lower()
        api_elements = ["路由", "模型", "验证", "错误", "文档"]
        found_elements = sum(1 for element in api_elements if element in content)
        
        ratio = found_elements / len(api_elements)
        if ratio >= 0.8:
            return "优秀"
        elif ratio >= 0.6:
            return "良好"
        else:
            return "基础"
    
    def _get_processing_info(self, control_result) -> Dict[str, Any]:
        """获取处理信息摘要"""
        return {
            "constraint_violations": len([r for r in control_result.constraint_violations if r.status.value == 'failed']),
            "validation_results": len([r for r in control_result.validation_results if r.status.value == 'passed']),
            "processing_results": len([r for r in control_result.processing_results if r.changes_made]),
            "re_prompt_attempts": control_result.re_prompt_attempts,
            "processing_time": control_result.processing_time
        }


async def demo_code_assistant():
    """代码助手演示"""
    print("💻 代码助手AgentWorker演示")
    print("=" * 50)
    
    try:
        # 初始化工作器
        worker = CodeAssistantWorker()
        
        # 演示1：生成函数
        print("\n🔧 演示1：生成函数")
        function_result = await worker.generate_function(
            function_description="实现一个计算斐波那契数列第n项的函数，支持缓存优化",
            language="python",
            include_tests=True
        )
        
        print(f"✅ 函数生成{'成功' if function_result['success'] else '失败'}")
        print(f"🏆 质量评分: {function_result['quality_score']:.1f}/10")
        print(f"🔒 安全评分: {function_result['security_score']:.1f}/10")
        print(f"⏱️ 处理时间: {function_result['processing_time']:.2f}s")
        
        # 演示2：生成类
        print("\n🔧 演示2：生成类")
        class_result = await worker.generate_class(
            class_description="设计一个简单的银行账户类，支持存款、取款、查询余额等操作",
            language="python",
            include_examples=True
        )
        
        print(f"✅ 类生成{'成功' if class_result['success'] else '失败'}")
        print(f"🏆 质量评分: {class_result['quality_score']:.1f}/10")
        print(f"🎨 设计质量: {class_result['design_quality']}")
        
        # 演示3：代码优化
        print("\n🔧 演示3：代码优化")
        sample_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
        
        optimize_result = await worker.optimize_code(
            code=sample_code,
            optimization_goals=["性能", "可读性"]
        )
        
        print(f"✅ 代码优化{'成功' if optimize_result['success'] else '失败'}")
        print(f"🏆 质量评分: {optimize_result['quality_score']:.1f}/10")
        print(f"⚡ 优化效果: {optimize_result['optimization_effectiveness']}")
        
        # 演示4：代码解释
        print("\n🔧 演示4：代码解释")
        explain_result = await worker.explain_code(
            code=sample_code,
            detail_level="intermediate"
        )
        
        print(f"✅ 代码解释{'成功' if explain_result['success'] else '失败'}")
        print(f"📖 解释质量: {explain_result['explanation_quality']}")
        
        # 演示5：代码审查
        print("\n🔧 演示5：代码审查")
        review_result = await worker.review_code(
            code=sample_code,
            review_focus=["代码质量", "性能", "可维护性"]
        )
        
        print(f"✅ 代码审查{'成功' if review_result['success'] else '失败'}")
        print(f"📋 审查质量: {review_result['review_quality']}")
        if review_result['security_issues']:
            print(f"⚠️ 安全问题: {len(review_result['security_issues'])}个")
        
        print("\n🎉 代码助手演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_code_assistant())