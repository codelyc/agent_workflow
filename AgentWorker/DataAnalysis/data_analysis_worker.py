"""
数据分析AgentWorker

专门用于数据报告、趋势分析、对比分析等数据处理任务的Agent工作流
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# 添加项目根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from core.output_control import OutputController
from core.config.llm_manager import LLMManager


class DataAnalysisWorker:
    """数据分析工作器"""
    
    def __init__(self, llm_model: str = "gpt-4o"):
        """
        初始化数据分析工作器
        
        Args:
            llm_model: 使用的LLM模型，数据分析推荐使用更强的模型
        """
        self.llm_model = llm_model
        self.llm_manager = LLMManager()
        
        # 从本地配置加载输出控制器
        config_path = Path(__file__).parent / "configs" / "output_control.yaml"
        self.output_controller = OutputController.from_config_file(
            str(config_path),
            llm_client=self.llm_manager.get_client()
        )
        
        print(f"✅ 数据分析工作器初始化完成，使用模型: {llm_model}")
    
    async def generate_data_report(self, data_description: str, analysis_focus: str, report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        生成数据分析报告
        
        Args:
            data_description: 数据描述
            analysis_focus: 分析重点
            report_type: 报告类型 (comprehensive/summary/detailed)
            
        Returns:
            包含分析报告和质量信息的字典
        """
        report_templates = {
            "comprehensive": "包含执行摘要、数据概述、详细分析、结论建议的完整报告",
            "summary": "重点突出关键发现和主要结论的简要报告",
            "detailed": "深入分析各个维度，提供详细数据支撑的专业报告"
        }
        
        prompt = f"""请基于以下数据信息生成数据分析报告：

数据描述：{data_description}
分析重点：{analysis_focus}
报告类型：{report_templates.get(report_type, '综合报告')}

报告要求：
1. 数据概述：简要描述数据的基本情况
2. 关键发现：识别数据中的重要模式、趋势和异常
3. 分析结论：基于数据得出客观、准确的结论
4. 行动建议：提供基于分析结果的实用建议
5. 使用结构化格式（Markdown）
6. 确保分析逻辑清晰，结论有数据支撑

请保持客观性，避免主观臆测，所有结论都要有数据依据。"""

        # 调用LLM生成初始分析
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="report"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "data_description": data_description,
            "analysis_focus": analysis_focus,
            "report_type": report_type,
            "quality_score": self._calculate_analysis_quality(control_result),
            "logical_consistency": self._check_logical_consistency(control_result),
            "processing_time": control_result.processing_time,
            "processing_info": self._get_processing_info(control_result)
        }
    
    async def perform_trend_analysis(self, time_series_data: str, time_period: str, prediction_horizon: str = "3个月") -> Dict[str, Any]:
        """
        执行趋势分析
        
        Args:
            time_series_data: 时间序列数据描述
            time_period: 分析时间段
            prediction_horizon: 预测时间范围
            
        Returns:
            包含趋势分析结果的字典
        """
        prompt = f"""请对以下时间序列数据进行趋势分析：

数据信息：{time_series_data}
分析时间段：{time_period}
预测时间范围：{prediction_horizon}

分析要求：
1. 历史趋势分析：识别数据的历史发展模式
2. 当前状态评估：分析当前数据状态和特征
3. 未来趋势预测：基于历史模式预测未来发展趋势
4. 影响因素分析：识别可能影响趋势的关键因素
5. 风险提示：指出趋势分析中的不确定性和风险

请确保分析基于数据规律，预测要合理谨慎。"""

        # 调用LLM生成分析
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="trend_analysis"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "time_series_data": time_series_data,
            "time_period": time_period,
            "prediction_horizon": prediction_horizon,
            "quality_score": self._calculate_analysis_quality(control_result),
            "trend_reliability": self._assess_trend_reliability(control_result)
        }
    
    async def perform_comparison_analysis(self, comparison_data: List[Dict], comparison_dimensions: List[str]) -> Dict[str, Any]:
        """
        执行对比分析
        
        Args:
            comparison_data: 对比数据列表
            comparison_dimensions: 对比维度列表
            
        Returns:
            包含对比分析结果的字典
        """
        data_summary = "\n".join([f"数据组{i+1}: {data}" for i, data in enumerate(comparison_data)])
        dimensions_text = "、".join(comparison_dimensions)
        
        prompt = f"""请对以下数据进行多维度对比分析：

对比数据：
{data_summary}

对比维度：{dimensions_text}

分析要求：
1. 对比维度分析：逐一分析各个维度的差异
2. 差异原因探讨：分析造成差异的可能原因
3. 优劣势评估：客观评估各数据组的优劣势
4. 关键洞察：提炼对比分析中的关键发现
5. 决策建议：基于对比结果提供决策参考

请保持客观中立，基于数据进行分析。"""

        # 调用LLM生成分析
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="comparison"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "comparison_data": comparison_data,
            "comparison_dimensions": comparison_dimensions,
            "quality_score": self._calculate_analysis_quality(control_result),
            "comparison_completeness": self._check_comparison_completeness(control_result)
        }
    
    async def generate_insights_summary(self, raw_data: str, business_context: str) -> Dict[str, Any]:
        """
        生成数据洞察摘要
        
        Args:
            raw_data: 原始数据描述
            business_context: 业务上下文
            
        Returns:
            包含洞察摘要的字典
        """
        prompt = f"""请基于以下原始数据生成业务洞察摘要：

原始数据：{raw_data}
业务背景：{business_context}

洞察要求：
1. 关键数据发现：提炼最重要的数据发现
2. 业务影响分析：分析数据对业务的潜在影响
3. 机会识别：识别数据中显示的业务机会
4. 风险警示：指出数据中反映的潜在风险
5. 行动建议：提供具体可行的行动建议

请确保洞察具有实际业务价值，建议可操作性强。"""

        # 调用LLM生成洞察
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制（使用通用report类型）
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="report"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "raw_data": raw_data,
            "business_context": business_context,
            "quality_score": self._calculate_analysis_quality(control_result),
            "insight_value": self._assess_insight_value(control_result)
        }
    
    def _calculate_analysis_quality(self, control_result) -> float:
        """计算分析质量分数"""
        if not control_result.success:
            return 0.0
        
        # 基础分数
        base_score = 8.0  # 数据分析要求更高的基础质量
        
        # 检查约束违反情况
        critical_violations = [r for r in control_result.constraint_violations 
                             if r.status.value == 'failed' and 'accuracy' in r.rule_name.lower()]
        if not critical_violations:
            base_score += 1.0
        
        # 检查AI自检结果
        ai_results = [r for r in control_result.validation_results 
                     if r.rule_name == 'ai_self_check' and r.details]
        if ai_results:
            ai_details = ai_results[0].details.get('ai_check_results', [])
            for check in ai_details:
                if check.get('type') == 'analysis_quality':
                    logic_score = check.get('logic', 0)
                    accuracy_score = check.get('accuracy', 0)
                    # 逻辑性和准确性权重更高
                    weighted_score = (logic_score * 0.4 + accuracy_score * 0.4) / 10 * 2
                    base_score = max(base_score, base_score + weighted_score - 1.0)
        
        # re-prompt次数影响
        if control_result.re_prompt_attempts > 0:
            base_score -= 0.3 * control_result.re_prompt_attempts
        
        return max(0.0, min(10.0, base_score))
    
    def _check_logical_consistency(self, control_result) -> bool:
        """检查逻辑一致性"""
        # 检查是否有逻辑矛盾相关的验证失败
        logic_failures = [r for r in control_result.validation_results 
                         if 'consistency' in r.rule_name.lower() and r.status.value == 'failed']
        return len(logic_failures) == 0
    
    def _assess_trend_reliability(self, control_result) -> str:
        """评估趋势分析可靠性"""
        if not control_result.success:
            return "低"
        
        # 基于质量分数评估可靠性
        quality = self._calculate_analysis_quality(control_result)
        if quality >= 8.5:
            return "高"
        elif quality >= 7.0:
            return "中"
        else:
            return "低"
    
    def _check_comparison_completeness(self, control_result) -> float:
        """检查对比分析完整性"""
        if not control_result.success:
            return 0.0
        
        # 检查是否包含必要的对比要素
        content = control_result.final_output.lower()
        required_elements = ['对比', '差异', '优势', '劣势', '建议']
        found_elements = sum(1 for element in required_elements if element in content)
        
        return found_elements / len(required_elements)
    
    def _assess_insight_value(self, control_result) -> str:
        """评估洞察价值"""
        if not control_result.success:
            return "低"
        
        content = control_result.final_output.lower()
        value_indicators = ['机会', '风险', '建议', '行动', '影响', '发现']
        found_indicators = sum(1 for indicator in value_indicators if indicator in content)
        
        value_ratio = found_indicators / len(value_indicators)
        if value_ratio >= 0.7:
            return "高"
        elif value_ratio >= 0.5:
            return "中"
        else:
            return "低"
    
    def _get_processing_info(self, control_result) -> Dict[str, Any]:
        """获取处理信息摘要"""
        return {
            "constraint_violations": len([r for r in control_result.constraint_violations if r.status.value == 'failed']),
            "validation_results": len([r for r in control_result.validation_results if r.status.value == 'passed']),
            "processing_results": len([r for r in control_result.processing_results if r.changes_made]),
            "re_prompt_attempts": control_result.re_prompt_attempts,
            "processing_time": control_result.processing_time
        }
    
    async def batch_analyze(self, analysis_tasks: List[Dict]) -> Dict[str, Any]:
        """批量数据分析"""
        results = []
        
        for task in analysis_tasks:
            task_type = task.get('type', 'report')
            
            try:
                if task_type == 'report':
                    result = await self.generate_data_report(
                        data_description=task['data_description'],
                        analysis_focus=task['analysis_focus'],
                        report_type=task.get('report_type', 'comprehensive')
                    )
                elif task_type == 'trend':
                    result = await self.perform_trend_analysis(
                        time_series_data=task['time_series_data'],
                        time_period=task['time_period'],
                        prediction_horizon=task.get('prediction_horizon', '3个月')
                    )
                elif task_type == 'comparison':
                    result = await self.perform_comparison_analysis(
                        comparison_data=task['comparison_data'],
                        comparison_dimensions=task['comparison_dimensions']
                    )
                elif task_type == 'insights':
                    result = await self.generate_insights_summary(
                        raw_data=task['raw_data'],
                        business_context=task['business_context']
                    )
                else:
                    result = {"success": False, "error": f"不支持的分析类型: {task_type}"}
                
                results.append(result)
                
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return {
            "total_tasks": len(analysis_tasks),
            "successful_tasks": sum(1 for r in results if r.get('success', False)),
            "average_quality": sum(r.get('quality_score', 0) for r in results) / len(results) if results else 0,
            "results": results
        }


async def demo_data_analysis():
    """数据分析演示"""
    print("📊 数据分析AgentWorker演示")
    print("=" * 50)
    
    try:
        # 初始化工作器
        worker = DataAnalysisWorker()
        
        # 演示1：数据报告生成
        print("\n📈 演示1：生成数据分析报告")
        report_result = await worker.generate_data_report(
            data_description="2024年Q1-Q3季度销售数据，包含产品类别、地区分布、客户群体等维度",
            analysis_focus="季度销售趋势和产品表现分析",
            report_type="comprehensive"
        )
        
        print(f"✅ 报告生成{'成功' if report_result['success'] else '失败'}")
        print(f"🏆 质量评分: {report_result['quality_score']:.1f}/10")
        print(f"🧠 逻辑一致性: {'是' if report_result['logical_consistency'] else '否'}")
        print(f"⏱️ 处理时间: {report_result['processing_time']:.2f}s")
        
        # 演示2：趋势分析
        print("\n📈 演示2：执行趋势分析")
        trend_result = await worker.perform_trend_analysis(
            time_series_data="过去12个月的网站访问量数据，包含月度访问量、用户来源、页面停留时间等指标",
            time_period="2024年1月-12月",
            prediction_horizon="未来6个月"
        )
        
        print(f"✅ 趋势分析{'成功' if trend_result['success'] else '失败'}")
        print(f"🏆 质量评分: {trend_result['quality_score']:.1f}/10")
        print(f"🎯 趋势可靠性: {trend_result['trend_reliability']}")
        
        # 演示3：对比分析
        print("\n📈 演示3：执行对比分析")
        comparison_result = await worker.perform_comparison_analysis(
            comparison_data=[
                {"name": "产品A", "sales": "100万", "growth": "15%", "satisfaction": "85%"},
                {"name": "产品B", "sales": "80万", "growth": "25%", "satisfaction": "90%"},
                {"name": "产品C", "sales": "120万", "growth": "5%", "satisfaction": "78%"}
            ],
            comparison_dimensions=["销售额", "增长率", "客户满意度", "市场潜力"]
        )
        
        print(f"✅ 对比分析{'成功' if comparison_result['success'] else '失败'}")
        print(f"🏆 质量评分: {comparison_result['quality_score']:.1f}/10")
        print(f"📋 分析完整性: {comparison_result['comparison_completeness']:.1%}")
        
        # 演示4：洞察摘要
        print("\n📈 演示4：生成洞察摘要")
        insights_result = await worker.generate_insights_summary(
            raw_data="用户行为数据显示：移动端访问占70%，购买转化率为3.2%，平均订单金额为268元",
            business_context="电商平台希望提升用户体验和销售转化"
        )
        
        print(f"✅ 洞察生成{'成功' if insights_result['success'] else '失败'}")
        print(f"🏆 质量评分: {insights_result['quality_score']:.1f}/10")
        print(f"💎 洞察价值: {insights_result['insight_value']}")
        
        print("\n🎉 数据分析演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_data_analysis())