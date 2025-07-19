"""
内容生成AgentWorker

专门用于文章、博客、营销文案等内容生成的Agent工作流
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from core.output_control import OutputController
from core.config.llm_manager import LLMManager


class ContentGenerationWorker:
    """内容生成工作器"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        初始化内容生成工作器
        
        Args:
            llm_model: 使用的LLM模型
        """
        self.llm_model = llm_model
        self.llm_manager = LLMManager()
        
        # 从本地配置加载输出控制器
        config_path = Path(__file__).parent / "configs" / "output_control.yaml"
        self.output_controller = OutputController.from_config_file(
            str(config_path),
            llm_client=self.llm_manager.get_client()
        )
        
        print(f"✅ 内容生成工作器初始化完成，使用模型: {llm_model}")
    
    async def generate_article(self, topic: str, target_audience: str = "通用读者", length: str = "中等") -> Dict[str, Any]:
        """
        生成文章
        
        Args:
            topic: 文章主题
            target_audience: 目标读者
            length: 文章长度 (短/中等/长)
            
        Returns:
            包含文章内容和质量信息的字典
        """
        # 构建prompt
        length_guide = {
            "短": "500-800字",
            "中等": "1000-1500字", 
            "长": "2000-3000字"
        }
        
        prompt = f"""请为主题"{topic}"撰写一篇面向{target_audience}的文章，长度约{length_guide.get(length, '1000-1500字')}。

要求：
1. 包含吸引人的标题
2. 结构完整：引言、主体内容、结论
3. 语言流畅，逻辑清晰
4. 提供实用价值
5. 使用Markdown格式

请确保内容原创、准确，避免敏感话题。"""

        # 调用LLM生成初始内容
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="article"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "original_content": control_result.original_output,
            "quality_score": self._calculate_quality_score(control_result),
            "improvements_made": len([r for r in control_result.processing_results if r.changes_made]),
            "re_prompt_attempts": control_result.re_prompt_attempts,
            "processing_time": control_result.processing_time,
            "topic": topic,
            "target_audience": target_audience,
            "length": length
        }
    
    async def generate_blog_post(self, topic: str, style: str = "conversational") -> Dict[str, Any]:
        """
        生成博客文章
        
        Args:
            topic: 博客主题
            style: 写作风格 (conversational/professional/casual)
            
        Returns:
            包含博客内容和质量信息的字典
        """
        style_guides = {
            "conversational": "采用对话式语调，亲切自然，如同与朋友交谈",
            "professional": "使用专业、正式的语言，注重权威性",
            "casual": "轻松随意的语调，幽默风趣"
        }
        
        prompt = f"""请撰写一篇关于"{topic}"的博客文章。

写作风格：{style_guides.get(style, '对话式语调')}

要求：
1. 包含个人观点和经验分享
2. 结构：引入话题、展开讨论、个人思考、结论
3. 长度控制在800-1500字
4. 使用Markdown格式
5. 可以包含一些个人化的元素

请确保内容有趣、有价值，能够引起读者共鸣。"""

        # 调用LLM生成内容
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="blog"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "quality_score": self._calculate_quality_score(control_result),
            "style": style,
            "topic": topic,
            "processing_info": self._get_processing_info(control_result)
        }
    
    async def generate_marketing_copy(self, product: str, target_market: str, copy_type: str = "销售页面") -> Dict[str, Any]:
        """
        生成营销文案
        
        Args:
            product: 产品名称
            target_market: 目标市场
            copy_type: 文案类型 (销售页面/广告文案/邮件营销)
            
        Returns:
            包含营销文案和效果评估的字典
        """
        copy_templates = {
            "销售页面": "包含价值主张、特色功能、客户证言、行动召唤的完整销售页面",
            "广告文案": "简洁有力的广告文案，突出核心卖点",
            "邮件营销": "个性化的邮件营销内容，包含主题行和正文"
        }
        
        prompt = f"""请为产品"{product}"创作{copy_type}，目标市场是{target_market}。

文案类型：{copy_templates.get(copy_type, '销售文案')}

要求：
1. 突出产品的核心价值和独特卖点
2. 针对目标市场的痛点和需求
3. 包含强有力的行动召唤
4. 语言有说服力，能激发购买欲望
5. 使用Markdown格式

请确保文案真实可信，避免夸大宣传。"""

        # 调用LLM生成内容
        llm_client = self.llm_manager.get_client()
        initial_output = await llm_client.generate(prompt, model=self.llm_model)
        
        # 应用输出控制
        control_result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=initial_output,
            output_type="marketing"
        )
        
        return {
            "success": control_result.success,
            "content": control_result.final_output,
            "product": product,
            "target_market": target_market,
            "copy_type": copy_type,
            "quality_score": self._calculate_quality_score(control_result),
            "processing_info": self._get_processing_info(control_result)
        }
    
    def _calculate_quality_score(self, control_result) -> float:
        """计算内容质量分数"""
        if not control_result.success:
            return 0.0
        
        # 基础分数
        base_score = 7.0
        
        # 如果没有约束违反，加分
        if not any(r.status.value == 'failed' for r in control_result.constraint_violations):
            base_score += 1.0
        
        # 如果验证全部通过，加分
        if all(r.status.value == 'passed' for r in control_result.validation_results):
            base_score += 1.0
        
        # 如果有处理改进，加分
        if any(r.changes_made for r in control_result.processing_results):
            base_score += 0.5
        
        # 如果需要re-prompt，减分
        if control_result.re_prompt_attempts > 0:
            base_score -= 0.5 * control_result.re_prompt_attempts
        
        return max(0.0, min(10.0, base_score))
    
    def _get_processing_info(self, control_result) -> Dict[str, Any]:
        """获取处理信息摘要"""
        return {
            "constraint_violations": len([r for r in control_result.constraint_violations if r.status.value == 'failed']),
            "validation_results": len([r for r in control_result.validation_results if r.status.value == 'passed']),
            "processing_results": len([r for r in control_result.processing_results if r.changes_made]),
            "re_prompt_attempts": control_result.re_prompt_attempts,
            "processing_time": control_result.processing_time
        }
    
    async def batch_generate(self, tasks: list) -> Dict[str, Any]:
        """批量生成内容"""
        results = []
        
        for task in tasks:
            task_type = task.get('type', 'article')
            
            if task_type == 'article':
                result = await self.generate_article(
                    topic=task['topic'],
                    target_audience=task.get('target_audience', '通用读者'),
                    length=task.get('length', '中等')
                )
            elif task_type == 'blog':
                result = await self.generate_blog_post(
                    topic=task['topic'],
                    style=task.get('style', 'conversational')
                )
            elif task_type == 'marketing':
                result = await self.generate_marketing_copy(
                    product=task['product'],
                    target_market=task['target_market'],
                    copy_type=task.get('copy_type', '销售页面')
                )
            else:
                result = {"success": False, "error": f"不支持的任务类型: {task_type}"}
            
            results.append(result)
        
        return {
            "total_tasks": len(tasks),
            "successful_tasks": sum(1 for r in results if r.get('success', False)),
            "average_quality": sum(r.get('quality_score', 0) for r in results) / len(results) if results else 0,
            "results": results
        }


async def demo_content_generation():
    """内容生成演示"""
    print("🎨 内容生成AgentWorker演示")
    print("=" * 50)
    
    try:
        # 初始化工作器
        worker = ContentGenerationWorker()
        
        # 演示1：生成文章
        print("\n📝 演示1：生成文章")
        article_result = await worker.generate_article(
            topic="人工智能在教育领域的应用前景",
            target_audience="教育工作者",
            length="中等"
        )
        
        print(f"✅ 文章生成{'成功' if article_result['success'] else '失败'}")
        print(f"🏆 质量评分: {article_result['quality_score']:.1f}/10")
        print(f"🔄 改进次数: {article_result['improvements_made']}")
        print(f"⏱️ 处理时间: {article_result['processing_time']:.2f}s")
        print(f"📄 内容长度: {len(article_result['content'])} 字符")
        
        # 演示2：生成博客
        print("\n📝 演示2：生成博客文章")
        blog_result = await worker.generate_blog_post(
            topic="远程工作的优缺点及个人体验",
            style="conversational"
        )
        
        print(f"✅ 博客生成{'成功' if blog_result['success'] else '失败'}")
        print(f"🏆 质量评分: {blog_result['quality_score']:.1f}/10")
        
        # 演示3：生成营销文案
        print("\n📝 演示3：生成营销文案")
        marketing_result = await worker.generate_marketing_copy(
            product="智能语言学习APP",
            target_market="年轻白领",
            copy_type="销售页面"
        )
        
        print(f"✅ 营销文案生成{'成功' if marketing_result['success'] else '失败'}")
        print(f"🏆 质量评分: {marketing_result['quality_score']:.1f}/10")
        
        # 演示4：批量生成
        print("\n📝 演示4：批量生成")
        batch_tasks = [
            {"type": "article", "topic": "可持续发展的重要性", "length": "短"},
            {"type": "blog", "topic": "健康生活小贴士", "style": "casual"},
            {"type": "marketing", "product": "有机咖啡", "target_market": "健康意识消费者"}
        ]
        
        batch_result = await worker.batch_generate(batch_tasks)
        print(f"✅ 批量生成完成: {batch_result['successful_tasks']}/{batch_result['total_tasks']}")
        print(f"🏆 平均质量: {batch_result['average_quality']:.1f}/10")
        
        print("\n🎉 内容生成演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_content_generation())