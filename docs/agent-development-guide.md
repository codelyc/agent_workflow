# 代理开发指南

本指南详细介绍如何在AI Agents Workflow框架中开发、配置和使用各种类型的代理。

## 目录

- [代理架构概述](#代理架构概述)
- [内置代理类型](#内置代理类型)
- [创建自定义代理](#创建自定义代理)
- [代理配置](#代理配置)
- [任务处理流程](#任务处理流程)
- [最佳实践](#最佳实践)

## 代理架构概述

### 核心组件

```
Agent Architecture
├── MicroAgent (基础代理)
│   ├── 任务处理逻辑
│   ├── LLM客户端
│   ├── 输出控制
│   └── 配置管理
├── SupervisorAgent (监督代理)
│   ├── 任务分发
│   ├── 结果聚合
│   └── 质量监控
└── AgentFactory (代理工厂)
    ├── 代理创建
    ├── 类型注册
    └── 配置验证
```

### 代理生命周期

1. **初始化阶段**
   - 加载配置
   - 初始化LLM客户端
   - 设置输出控制器

2. **任务接收阶段**
   - 解析任务定义
   - 验证任务参数
   - 准备执行环境

3. **任务执行阶段**
   - 调用LLM生成内容
   - 应用约束条件
   - 执行后处理逻辑

4. **结果输出阶段**
   - 质量评估
   - 格式化输出
   - 记录执行日志

## 内置代理类型

### 1. 代码助手代理 (CodeAssistant)

**功能特性：**
- 代码生成和优化
- 代码审查和解释
- 类和函数设计
- 代码重构建议

**使用示例：**
```python
from AgentWorker.CodeAssistant.code_assistant_worker import CodeAssistantWorker

# 创建代理实例
code_agent = CodeAssistantWorker()

# 代码生成任务
result = await code_agent.process_task({
    "type": "code_generation",
    "prompt": "创建一个计算斐波那契数列的函数",
    "language": "python",
    "constraints": {
        "max_lines": 20,
        "include_docstring": True
    }
})
```

**配置选项：**
```yaml
code_assistant:
  programming_languages: ["python", "javascript", "java"]
  code_style: "pep8"
  include_tests: true
  optimization_level: "standard"
```

### 2. 内容生成代理 (ContentGeneration)

**功能特性：**
- 文章和博客写作
- 营销文案生成
- 技术文档编写
- 创意内容创作

**使用示例：**
```python
from AgentWorker.ContentGeneration.content_generation_worker import ContentGenerationWorker

# 创建代理实例
content_agent = ContentGenerationWorker()

# 内容生成任务
result = await content_agent.process_task({
    "type": "article_writing",
    "topic": "人工智能在教育中的应用",
    "target_audience": "教育工作者",
    "constraints": {
        "word_count": 1500,
        "tone": "professional",
        "include_examples": True
    }
})
```

**配置选项：**
```yaml
content_generation:
  default_tone: "professional"
  supported_formats: ["markdown", "html", "plain_text"]
  seo_optimization: true
  plagiarism_check: true
```

### 3. 数据分析代理 (DataAnalysis)

**功能特性：**
- 数据报告生成
- 趋势分析
- 统计计算
- 可视化建议

**使用示例：**
```python
from AgentWorker.DataAnalysis.data_analysis_worker import DataAnalysisWorker

# 创建代理实例
data_agent = DataAnalysisWorker()

# 数据分析任务
result = await data_agent.process_task({
    "type": "trend_analysis",
    "data_source": "sales_data.csv",
    "analysis_period": "last_quarter",
    "constraints": {
        "include_charts": True,
        "confidence_level": 0.95,
        "report_format": "executive_summary"
    }
})
```

**配置选项：**
```yaml
data_analysis:
  supported_formats: ["csv", "json", "excel"]
  visualization_library: "matplotlib"
  statistical_methods: ["regression", "correlation", "clustering"]
  export_formats: ["pdf", "html", "png"]
```

## 创建自定义代理

### 步骤1：定义代理类

```python
from src.agents.micro.base_agent import MicroAgent
from src.core.config.agent_config import AgentConfig
from src.core.task_types.task_definition import TaskDefinition
from typing import Dict, Any

class CustomTranslationAgent(MicroAgent):
    """自定义翻译代理"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.supported_languages = config.get('supported_languages', ['en', 'zh', 'es', 'fr'])
        self.translation_quality = config.get('translation_quality', 'high')
    
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """处理翻译任务"""
        
        # 1. 验证任务参数
        source_text = task.get_parameter('source_text')
        source_lang = task.get_parameter('source_language')
        target_lang = task.get_parameter('target_language')
        
        if not self._validate_languages(source_lang, target_lang):
            raise ValueError(f"不支持的语言对: {source_lang} -> {target_lang}")
        
        # 2. 构建翻译提示
        prompt = self._build_translation_prompt(
            source_text, source_lang, target_lang
        )
        
        # 3. 调用LLM进行翻译
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,  # 翻译任务使用较低的温度
            max_tokens=len(source_text) * 2  # 动态设置最大token数
        )
        
        # 4. 后处理和质量检查
        translated_text = self._post_process_translation(response.content)
        quality_score = await self._assess_translation_quality(
            source_text, translated_text, source_lang, target_lang
        )
        
        # 5. 返回结果
        return {
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang,
            'quality_score': quality_score,
            'word_count': len(translated_text.split()),
            'processing_info': self._get_processing_info()
        }
    
    def _validate_languages(self, source_lang: str, target_lang: str) -> bool:
        """验证语言支持"""
        return (source_lang in self.supported_languages and 
                target_lang in self.supported_languages and 
                source_lang != target_lang)
    
    def _build_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """构建翻译提示"""
        return f"""
        请将以下{source_lang}文本翻译成{target_lang}：
        
        原文：
        {text}
        
        翻译要求：
        1. 保持原文的语气和风格
        2. 确保术语翻译的准确性
        3. 保持格式和结构
        4. 如有文化差异，请适当本地化
        
        翻译：
        """
    
    def _post_process_translation(self, translation: str) -> str:
        """后处理翻译结果"""
        # 清理格式
        translation = translation.strip()
        # 移除可能的提示词残留
        if translation.startswith('翻译：'):
            translation = translation[3:].strip()
        return translation
    
    async def _assess_translation_quality(self, source: str, translation: str, 
                                        source_lang: str, target_lang: str) -> float:
        """评估翻译质量"""
        # 这里可以实现更复杂的质量评估逻辑
        # 例如：BLEU分数、语义相似度等
        
        # 简单的长度比例检查
        length_ratio = len(translation) / len(source)
        if 0.5 <= length_ratio <= 2.0:
            return 0.8
        else:
            return 0.6
    
    def _get_processing_info(self) -> Dict[str, Any]:
        """获取处理信息"""
        return {
            'agent_type': 'translation',
            'model_used': self.config.model,
            'processing_time': '2.3s',
            'tokens_used': 150
        }
```

### 步骤2：注册代理类型

```python
# 在 src/agents/factory/agent_factory.py 中注册
from .custom_translation_agent import CustomTranslationAgent

class AgentFactory:
    _agent_types = {
        'code_assistant': CodeAssistantAgent,
        'content_generation': ContentGenerationAgent,
        'data_analysis': DataAnalysisAgent,
        'translation': CustomTranslationAgent,  # 新增
    }
```

### 步骤3：创建配置文件

```yaml
# configs/translation_agent_config.yaml
agent:
  type: "translation"
  llm_provider: "openai"
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2048
  
translation:
  supported_languages: ["en", "zh", "es", "fr", "de", "ja"]
  translation_quality: "high"
  enable_context_awareness: true
  preserve_formatting: true
  
output_control:
  quality_threshold: 0.7
  max_retries: 2
  validation_rules:
    - "length_check"
    - "language_detection"
    - "format_preservation"
```

### 步骤4：创建工作器

```python
# AgentWorker/Translation/translation_worker.py
import asyncio
from src.agents.factory.agent_factory import AgentFactory
from src.core.config.config_manager import ConfigManager
from src.core.task_types.task_definition import TaskDefinition

class TranslationWorker:
    def __init__(self):
        # 加载配置
        config_manager = ConfigManager()
        self.config = config_manager.load_config('translation_agent_config.yaml')
        
        # 创建代理
        self.agent = AgentFactory.create_agent('translation', self.config)
    
    async def translate_text(self, source_text: str, source_lang: str, target_lang: str):
        """翻译文本"""
        task = TaskDefinition(
            task_type='translation',
            parameters={
                'source_text': source_text,
                'source_language': source_lang,
                'target_language': target_lang
            }
        )
        
        result = await self.agent.process_task(task)
        return result
    
    async def batch_translate(self, texts: list, source_lang: str, target_lang: str):
        """批量翻译"""
        tasks = []
        for text in texts:
            task = self.translate_text(text, source_lang, target_lang)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

# 演示用法
async def main():
    worker = TranslationWorker()
    
    # 单个翻译
    result = await worker.translate_text(
        "Hello, how are you?", 
        "en", 
        "zh"
    )
    print(f"翻译结果: {result['translated_text']}")
    
    # 批量翻译
    texts = [
        "Good morning!",
        "How can I help you?",
        "Thank you very much."
    ]
    results = await worker.batch_translate(texts, "en", "zh")
    for i, result in enumerate(results):
        print(f"文本{i+1}: {result['translated_text']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 代理配置

### 基础配置结构

```yaml
# 代理基础配置
agent:
  type: "agent_type_name"          # 代理类型
  llm_provider: "openai"           # LLM提供商
  model: "gpt-4"                   # 模型名称
  temperature: 0.7                 # 生成温度
  max_tokens: 2048                 # 最大token数
  timeout: 30                      # 超时时间(秒)
  
# 输出控制配置
output_control:
  quality_threshold: 0.8           # 质量阈值
  max_retries: 3                   # 最大重试次数
  validation_enabled: true         # 启用验证
  validation_rules:                # 验证规则
    - "length_check"
    - "format_check"
    - "content_check"
  
# 日志配置
logging:
  level: "INFO"                    # 日志级别
  file_path: "logs/agent.log"      # 日志文件路径
  max_file_size: "10MB"            # 最大文件大小
  backup_count: 5                  # 备份文件数量
  
# 性能配置
performance:
  enable_caching: true             # 启用缓存
  cache_ttl: 3600                  # 缓存过期时间(秒)
  concurrent_tasks: 5              # 并发任务数
  memory_limit: "1GB"              # 内存限制
```

### 高级配置选项

```yaml
# 自定义提示模板
prompt_templates:
  system_prompt: |
    你是一个专业的{agent_type}代理。
    请根据用户的要求完成任务，确保输出质量和准确性。
  
  task_prompt: |
    任务类型: {task_type}
    任务描述: {task_description}
    约束条件: {constraints}
    
    请完成以上任务。

# 工具配置
tools:
  enabled_tools:                   # 启用的工具
    - "web_search"
    - "file_operations"
    - "data_analysis"
  
  tool_configs:                    # 工具配置
    web_search:
      max_results: 10
      timeout: 15
    file_operations:
      allowed_extensions: [".txt", ".md", ".json"]
      max_file_size: "10MB"

# 安全配置
security:
  input_sanitization: true         # 输入清理
  output_filtering: true           # 输出过滤
  rate_limiting:
    requests_per_minute: 60
    burst_limit: 10
  
  content_policy:
    block_harmful_content: true
    block_personal_info: true
    block_copyrighted_content: true
```

## 任务处理流程

### 任务定义结构

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class TaskDefinition:
    """任务定义类"""
    task_type: str                   # 任务类型
    task_id: Optional[str] = None    # 任务ID
    priority: int = 1                # 优先级 (1-10)
    parameters: Dict[str, Any] = None # 任务参数
    constraints: Dict[str, Any] = None # 约束条件
    context: Dict[str, Any] = None   # 上下文信息
    metadata: Dict[str, Any] = None  # 元数据
    
    def get_parameter(self, key: str, default=None):
        """获取任务参数"""
        return self.parameters.get(key, default) if self.parameters else default
    
    def get_constraint(self, key: str, default=None):
        """获取约束条件"""
        return self.constraints.get(key, default) if self.constraints else default
```

### 处理流程示例

```python
async def process_task_with_monitoring(self, task: TaskDefinition) -> Dict[str, Any]:
    """带监控的任务处理流程"""
    
    # 1. 任务预处理
    start_time = time.time()
    task_id = task.task_id or self._generate_task_id()
    
    try:
        # 2. 输入验证
        self._validate_task_input(task)
        
        # 3. 上下文准备
        context = await self._prepare_context(task)
        
        # 4. 任务执行
        with self._create_execution_context(task_id):
            result = await self._execute_task_core(task, context)
        
        # 5. 结果验证
        validated_result = await self._validate_result(result, task)
        
        # 6. 后处理
        final_result = await self._post_process_result(validated_result, task)
        
        # 7. 记录成功
        execution_time = time.time() - start_time
        await self._log_task_success(task_id, execution_time, final_result)
        
        return final_result
        
    except Exception as e:
        # 8. 错误处理
        execution_time = time.time() - start_time
        await self._log_task_error(task_id, execution_time, e)
        
        # 9. 重试逻辑
        if self._should_retry(task, e):
            return await self._retry_task(task)
        else:
            raise
```

## 最佳实践

### 1. 代理设计原则

- **单一职责**: 每个代理专注于特定领域
- **可配置性**: 通过配置文件控制行为
- **可扩展性**: 支持插件和自定义扩展
- **容错性**: 优雅处理错误和异常
- **可监控性**: 提供详细的执行日志和指标

### 2. 性能优化

```python
# 使用缓存减少重复计算
from functools import lru_cache

class OptimizedAgent(MicroAgent):
    @lru_cache(maxsize=128)
    def _cached_preprocessing(self, input_text: str) -> str:
        """缓存预处理结果"""
        return self._expensive_preprocessing(input_text)
    
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        # 使用缓存的预处理结果
        preprocessed = self._cached_preprocessing(task.get_parameter('text'))
        # ... 其他处理逻辑
```

### 3. 错误处理策略

```python
class RobustAgent(MicroAgent):
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        max_retries = self.config.get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                result = await self._attempt_task(task)
                return result
            except TemporaryError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    raise
            except PermanentError as e:
                # 永久性错误不重试
                raise
```

### 4. 资源管理

```python
class ResourceManagedAgent(MicroAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._semaphore = asyncio.Semaphore(config.get('max_concurrent_tasks', 5))
    
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        async with self._semaphore:
            # 限制并发任务数量
            return await super().process_task(task)
```

### 5. 测试策略

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestCustomAgent:
    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            agent_type='custom',
            llm_provider='mock',
            model='test-model'
        )
    
    @pytest.fixture
    def mock_llm_client(self):
        client = AsyncMock()
        client.generate.return_value = MagicMock(content="测试输出")
        return client
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, agent_config, mock_llm_client):
        agent = CustomAgent(agent_config)
        agent.llm_client = mock_llm_client
        
        task = TaskDefinition(
            task_type='test',
            parameters={'input': '测试输入'}
        )
        
        result = await agent.process_task(task)
        
        assert result is not None
        assert 'output' in result
        mock_llm_client.generate.assert_called_once()
```

---

通过遵循本指南，您可以创建高质量、可维护的AI代理，并充分利用框架提供的各种功能和工具。