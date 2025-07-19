# Agent 输出控制系统

基于"前置约束+自检+后处理"思路设计的Agent输出控制框架，确保Agent输出始终稳定、可控。

## 🎯 核心思路

让Agent最终输出符合预期的核心在于"前置约束+自检+后处理"：

1. **前置约束**：设计阶段清晰定义输出目标和格式，在Prompt中加入硬性要求并多举例
2. **自检验证**：Agent输出后用程序或AI自身二次校验
3. **后处理**：必要时re-prompt修正，并可引入奖励模型辅助判别
4. **可配置化**：保障Agent输出始终稳定、可控

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│                 OutputController                │
├─────────────────────────────────────────────────┤
│  ┌───────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │ ConstraintEngine │ │ OutputValidator│ │Processor│ │
│  │  前置约束检查   │ │   自检验证    │ │ 后处理 │ │
│  └───────────────┘ └──────────────┘ └─────────┘ │
└─────────────────────────────────────────────────┘
                        │
                 ┌─────────────┐
                 │ Re-prompt   │
                 │  重新生成    │
                 └─────────────┘
```

## 🚀 快速开始

### 基本使用

```python
import asyncio
from src.output_control import OutputController

async def basic_usage():
    # 从配置文件创建控制器
    controller = OutputController.from_config_file(
        "configs/output_control.yaml",
        llm_client=your_llm_client
    )
    
    # 控制Agent输出
    result = await controller.control_output(
        original_prompt="请分析当前市场情况",
        agent_output="市场还不错",  # 原始输出
        output_type="analysis"
    )
    
    print(f"控制成功: {result.success}")
    print(f"最终输出: {result.final_output}")

asyncio.run(basic_usage())
```

### 与现有Agent集成

```python
from src.output_control import OutputController

class AgentWrapper:
    def __init__(self, base_agent, output_controller):
        self.base_agent = base_agent
        self.output_controller = output_controller
    
    async def execute_with_control(self, prompt, output_type=None):
        # 1. 增强prompt
        enhanced_prompt = self._add_constraints_to_prompt(prompt, output_type)
        
        # 2. 调用原始Agent
        agent_output = await self.base_agent.generate(enhanced_prompt)
        
        # 3. 应用输出控制
        result = await self.output_controller.control_output(
            original_prompt=prompt,
            agent_output=agent_output,
            output_type=output_type
        )
        
        return result.final_output
```

## 📋 配置说明

### 配置文件结构

```yaml
# 全局设置
global:
  enabled: true
  max_retry_attempts: 3
  debug_mode: false

# 前置约束
pre_constraints:
  # 格式约束
  format_constraints:
    - name: "json_format"
      description: "输出必须是有效的JSON格式"
      type: "format"
      required: true
      examples: ['{"status": "success"}']
  
  # 内容约束  
  content_constraints:
    - name: "no_sensitive_info"
      description: "禁止输出敏感信息"
      type: "content"
      blacklist_patterns: ["password", "api_key"]
  
  # 长度约束
  length_constraints:
    - name: "reasonable_length"
      description: "合理的输出长度"
      type: "length"
      min_length: 10
      max_length: 1000

# 验证配置
validation:
  # AI自检
  ai_self_check:
    enabled: true
    model: "gpt-4o-mini"
    check_prompts:
      consistency: "检查逻辑一致性..."
      quality_check: "评估输出质量..."

# 后处理配置
post_processing:
  # 文本处理
  text_processing:
    enabled: true
    strip_whitespace: true
    normalize_line_endings: true
  
  # 格式化
  formatting:
    enabled: true
    auto_format_json: true
    auto_format_markdown: true
  
  # 过滤
  filtering:
    enabled: true
    remove_sensitive_info: true

# Re-prompt配置
re_prompt:
  enabled: true
  max_attempts: 3
  strategy: "progressive"
  templates:
    format_fix: "你的输出格式不正确，请重新生成..."
```

## 🎛️ 核心组件

### 1. ConstraintEngine (前置约束引擎)

负责在输出生成前和生成后检查各种约束条件：

```python
from src.output_control.constraints import ConstraintEngine
from src.output_control.models import Constraint, ConstraintType

# 创建约束
constraint = Constraint(
    name="json_format",
    description="JSON格式约束",
    type=ConstraintType.FORMAT,
    pattern=r"^\{.*\}$",
    required=True
)

# 使用约束引擎
engine = ConstraintEngine([constraint])
violations = engine.check_constraints(output_text)
```

**支持的约束类型：**
- **格式约束**：JSON、XML、Markdown等格式验证
- **内容约束**：敏感信息过滤、必需元素检查
- **长度约束**：最小/最大长度限制
- **逻辑约束**：一致性检查、完整性验证

### 2. OutputValidator (自检验证器)

提供多层次的输出验证：

```python
from src.output_control.validator import OutputValidator

validator = OutputValidator(validation_rules, llm_client)
results = await validator.validate_output(output_text)
```

**验证类型：**
- **结构验证**：JSON Schema、XML格式验证
- **内容验证**：拼写检查、语法检查、事实检查
- **AI自检**：使用LLM进行一致性、格式、质量检查

### 3. OutputProcessor (后处理器)

对输出进行清理、格式化和改进：

```python
from src.output_control.processor import OutputProcessor

processor = OutputProcessor(processing_rules)
processed_output, results = await processor.process_output(output_text)
```

**处理类型：**
- **文本处理**：空白符处理、行结束符标准化
- **格式化**：JSON美化、Markdown格式化、代码块美化
- **补全**：添加缺失章节、生成摘要、添加元数据
- **过滤**：敏感信息移除、不当内容过滤

### 4. Re-prompt机制

当验证失败时自动重新生成：

```python
# 配置re-prompt模板
templates = {
    "format_fix": """你的回答格式不正确：
    
问题：{original_question}
格式要求：{format_requirements}
错误：{validation_errors}

请重新生成符合格式的回答。"""
}
```

## 🔧 高级功能

### 自定义约束

```python
from src.output_control.models import Constraint, ConstraintType

# 创建自定义约束
custom_constraint = Constraint(
    name="business_format",
    description="商业报告格式",
    type=ConstraintType.CONTENT,
    required_elements=["executive_summary", "analysis", "recommendations"],
    examples=["# Executive Summary\n...", "## Analysis\n..."]
)
```

### 输出类型特定配置

```yaml
output_types:
  # 代码输出
  code:
    pre_constraints:
      - name: "syntax_check"
        languages: ["python", "javascript"]
    post_processing:
      - name: "code_formatting"
        enabled: true
        
  # 分析报告
  analysis:
    pre_constraints:
      - name: "structure_requirement"
        sections: ["背景", "分析", "结论", "建议"]
        
  # JSON响应
  json:
    pre_constraints:
      - name: "json_schema"
        schema_validation: true
    post_processing:
      - name: "json_prettify"
        enabled: true
```

### 奖励模型集成

```yaml
reward_model:
  enabled: true
  model_type: "classification"
  threshold: 0.7
  evaluation_criteria:
    - name: "helpfulness"
      weight: 0.3
    - name: "accuracy" 
      weight: 0.3
    - name: "clarity"
      weight: 0.2
    - name: "completeness"
      weight: 0.2
```

## 📊 使用统计

系统提供详细的使用统计信息：

```python
# 获取控制摘要
summary = controller.get_control_summary(result)
print(f"成功率: {summary['success_rate']}")
print(f"Re-prompt次数: {summary['re_prompt_attempts']}")
print(f"处理改进: {summary['improvements_made']}")

# 获取包装器统计
wrapper = AgentWrapper(agent, controller)
stats = wrapper.get_stats()
print(f"总请求: {stats['total_requests']}")
print(f"成功率: {stats['success_rate']:.2%}")
```

## 🎨 使用场景

### 1. JSON API响应控制

```python
# 确保API总是返回有效JSON
config = {
    "pre_constraints": {
        "format_constraints": [{
            "name": "json_format",
            "required": True
        }]
    },
    "post_processing": {
        "formatting": {
            "auto_format_json": True
        }
    }
}
```

### 2. 文档生成质量控制

```python
# 确保生成的文档包含必要章节
config = {
    "pre_constraints": {
        "content_constraints": [{
            "required_elements": ["introduction", "methodology", "results", "conclusion"]
        }]
    },
    "post_processing": {
        "completion": {
            "add_missing_sections": True
        }
    }
}
```

### 3. 代码生成安全控制

```python
# 防止生成包含敏感信息的代码
config = {
    "pre_constraints": {
        "content_constraints": [{
            "blacklist_patterns": ["api_key", "password", "secret"]
        }]
    },
    "post_processing": {
        "filtering": {
            "remove_sensitive_info": True
        }
    }
}
```

## 🛠️ 开发指南

### 扩展约束类型

```python
from src.output_control.constraints import ConstraintEngine

class CustomConstraintEngine(ConstraintEngine):
    def _check_custom_constraint(self, output, constraint, context):
        # 实现自定义约束逻辑
        if custom_validation_failed:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                rule_name=constraint.name,
                message="自定义验证失败",
                suggestions=["修改建议"]
            )
        return ValidationResult(status=ValidationStatus.PASSED, ...)
```

### 添加新的验证器

```python
from src.output_control.validator import OutputValidator

class CustomValidator(OutputValidator):
    async def _custom_validation(self, output, rule, context):
        # 实现自定义验证逻辑
        return ValidationResult(...)
```

### 自定义后处理器

```python
from src.output_control.processor import OutputProcessor

class CustomProcessor(OutputProcessor):
    async def _custom_processing(self, output, rule, context):
        # 实现自定义处理逻辑
        processed_output = self._apply_custom_processing(output)
        return processed_output, ProcessingResult(...)
```

## 📈 性能优化

### 1. 批量处理优化

```python
# 禁用re-prompt以提高批量处理速度
config = OutputControlConfig(
    re_prompt_enabled=False,
    logging_enabled=False
)
```

### 2. 异步并发处理

```python
import asyncio

async def batch_control(controller, tasks):
    return await asyncio.gather(*[
        controller.control_output(prompt, output) 
        for prompt, output in tasks
    ])
```

### 3. 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_constraint_check(output_hash, constraints_hash):
    # 缓存约束检查结果
    pass
```

## 🔍 调试和监控

### 启用调试模式

```yaml
global:
  debug_mode: true
  save_intermediate_results: true

logging:
  enabled: true
  level: "DEBUG"
  log_file: "logs/output_control.log"
```

### 监控指标

```python
# 监控关键指标
metrics = {
    'constraint_violation_rate': failed_constraints / total_constraints,
    'validation_failure_rate': failed_validations / total_validations,
    're_prompt_rate': re_prompt_attempts / total_requests,
    'processing_improvement_rate': processed_improvements / total_requests,
    'average_processing_time': sum(times) / len(times)
}
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者。

---

**让每一个Agent的输出都可控、可靠、可预期！** 🎯 