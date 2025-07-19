# 配置指南

本指南详细介绍如何配置AI Agents Workflow框架的各个组件，包括代理配置、LLM设置、输出控制、工具配置等。

## 目录

- [配置文件结构](#配置文件结构)
- [代理配置](#代理配置)
- [LLM提供商配置](#llm提供商配置)
- [输出控制配置](#输出控制配置)
- [工具配置](#工具配置)
- [日志配置](#日志配置)
- [性能配置](#性能配置)
- [安全配置](#安全配置)
- [环境变量](#环境变量)
- [配置最佳实践](#配置最佳实践)

## 配置文件结构

### 基础配置文件格式

框架支持YAML和JSON两种配置文件格式。推荐使用YAML格式，因为它更易读和维护。

```yaml
# config.yaml - 基础配置文件结构

# 代理基础配置
agent:
  type: "code_assistant"              # 代理类型
  name: "MyCodeAssistant"             # 代理名称
  version: "1.0.0"                    # 版本号
  description: "专业的代码助手代理"    # 描述

# LLM配置
llm:
  provider: "openai"                  # LLM提供商
  model: "gpt-4"                      # 模型名称
  api_key: "${OPENAI_API_KEY}"        # API密钥（支持环境变量）
  base_url: "https://api.openai.com/v1" # API基础URL
  temperature: 0.7                    # 生成温度
  max_tokens: 2048                    # 最大token数
  timeout: 30                         # 请求超时时间

# 输出控制配置
output_control:
  enabled: true                       # 启用输出控制
  quality_threshold: 0.8              # 质量阈值
  max_retries: 3                      # 最大重试次数
  validation_rules:                   # 验证规则
    - "length_check"
    - "format_check"
    - "content_check"

# 日志配置
logging:
  level: "INFO"                       # 日志级别
  file_path: "logs/agent.log"         # 日志文件路径
  max_file_size: "10MB"               # 最大文件大小
  backup_count: 5                     # 备份文件数量
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 性能配置
performance:
  enable_caching: true                # 启用缓存
  cache_ttl: 3600                     # 缓存过期时间
  concurrent_tasks: 5                 # 并发任务数
  memory_limit: "1GB"                 # 内存限制

# 自定义配置
custom:
  # 在这里添加特定于代理的配置
  programming_languages: ["python", "javascript", "java"]
  code_style: "pep8"
  include_tests: true
```

### 配置文件组织

推荐的配置文件组织结构：

```
configs/
├── base/                           # 基础配置
│   ├── default.yaml               # 默认配置
│   ├── logging.yaml               # 日志配置
│   └── security.yaml              # 安全配置
├── agents/                         # 代理配置
│   ├── code_assistant.yaml        # 代码助手配置
│   ├── content_generation.yaml    # 内容生成配置
│   └── data_analysis.yaml         # 数据分析配置
├── llm/                           # LLM配置
│   ├── openai.yaml                # OpenAI配置
│   ├── anthropic.yaml             # Anthropic配置
│   └── local.yaml                 # 本地模型配置
├── environments/                   # 环境配置
│   ├── development.yaml           # 开发环境
│   ├── staging.yaml               # 测试环境
│   └── production.yaml            # 生产环境
└── workflows/                      # 工作流配置
    ├── sequential.yaml             # 顺序工作流
    └── parallel.yaml               # 并行工作流
```

## 代理配置

### 代码助手代理配置

```yaml
# configs/agents/code_assistant.yaml

agent:
  type: "code_assistant"
  name: "AdvancedCodeAssistant"
  version: "2.0.0"
  description: "高级代码助手，支持多种编程语言"

# LLM配置
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.3                    # 代码生成使用较低温度
  max_tokens: 4096
  top_p: 0.9
  frequency_penalty: 0.1
  presence_penalty: 0.1

# 代码助手特定配置
code_assistant:
  # 支持的编程语言
  supported_languages:
    - name: "python"
      extensions: [".py"]
      style_guide: "pep8"
      linter: "pylint"
    - name: "javascript"
      extensions: [".js", ".ts"]
      style_guide: "eslint"
      linter: "eslint"
    - name: "java"
      extensions: [".java"]
      style_guide: "google"
      linter: "checkstyle"
  
  # 代码生成配置
  generation:
    include_docstrings: true          # 包含文档字符串
    include_type_hints: true          # 包含类型提示
    include_tests: false              # 是否包含测试
    max_function_length: 50           # 最大函数长度
    max_class_methods: 20             # 最大类方法数
  
  # 代码审查配置
  review:
    check_style: true                 # 检查代码风格
    check_security: true              # 检查安全问题
    check_performance: true           # 检查性能问题
    check_maintainability: true       # 检查可维护性
    severity_levels: ["error", "warning", "info"]
  
  # 重构配置
  refactoring:
    extract_methods: true             # 提取方法
    rename_variables: true            # 重命名变量
    optimize_imports: true            # 优化导入
    remove_dead_code: true            # 移除死代码

# 输出控制配置
output_control:
  quality_threshold: 0.85
  validation_rules:
    - name: "syntax_check"
      enabled: true
      severity: "error"
    - name: "style_check"
      enabled: true
      severity: "warning"
    - name: "complexity_check"
      enabled: true
      max_complexity: 10
    - name: "length_check"
      enabled: true
      max_lines: 100
      max_characters: 5000

# 工具配置
tools:
  enabled_tools:
    - "syntax_validator"
    - "code_formatter"
    - "static_analyzer"
    - "test_generator"
  
  tool_configs:
    syntax_validator:
      timeout: 10
      languages: ["python", "javascript", "java"]
    code_formatter:
      style: "auto"
      line_length: 88
    static_analyzer:
      rules: ["security", "performance", "maintainability"]
    test_generator:
      framework: "pytest"
      coverage_threshold: 0.8
```

### 内容生成代理配置

```yaml
# configs/agents/content_generation.yaml

agent:
  type: "content_generation"
  name: "ContentCreator"
  version: "1.5.0"
  description: "专业内容生成代理"

# LLM配置
llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
  temperature: 0.8                    # 创意内容使用较高温度
  max_tokens: 8192
  top_p: 0.95

# 内容生成特定配置
content_generation:
  # 支持的内容类型
  content_types:
    - name: "article"
      min_words: 500
      max_words: 3000
      structure: ["introduction", "body", "conclusion"]
    - name: "blog_post"
      min_words: 300
      max_words: 1500
      seo_optimized: true
    - name: "marketing_copy"
      min_words: 50
      max_words: 500
      persuasive: true
    - name: "technical_doc"
      min_words: 1000
      max_words: 5000
      include_examples: true
  
  # 写作风格配置
  writing_styles:
    - name: "professional"
      tone: "formal"
      vocabulary: "advanced"
    - name: "casual"
      tone: "informal"
      vocabulary: "simple"
    - name: "academic"
      tone: "scholarly"
      vocabulary: "technical"
    - name: "creative"
      tone: "engaging"
      vocabulary: "varied"
  
  # SEO配置
  seo:
    enabled: true
    keyword_density: 0.02             # 关键词密度
    meta_description_length: 160      # 元描述长度
    title_length: 60                  # 标题长度
    heading_structure: true           # 标题结构
  
  # 质量控制
  quality_control:
    readability_score: 60             # 可读性评分
    grammar_check: true               # 语法检查
    plagiarism_check: false           # 抄袭检查
    fact_check: false                 # 事实检查

# 输出控制配置
output_control:
  quality_threshold: 0.75
  validation_rules:
    - name: "word_count_check"
      enabled: true
    - name: "readability_check"
      enabled: true
      min_score: 60
    - name: "grammar_check"
      enabled: true
    - name: "structure_check"
      enabled: true
```

### 数据分析代理配置

```yaml
# configs/agents/data_analysis.yaml

agent:
  type: "data_analysis"
  name: "DataAnalyst"
  version: "1.8.0"
  description: "专业数据分析代理"

# LLM配置
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.2                    # 数据分析使用低温度
  max_tokens: 6144

# 数据分析特定配置
data_analysis:
  # 支持的数据格式
  supported_formats:
    - name: "csv"
      max_size: "100MB"
      encoding: ["utf-8", "gbk"]
    - name: "json"
      max_size: "50MB"
      nested_levels: 5
    - name: "excel"
      max_size: "200MB"
      sheets: "all"
    - name: "parquet"
      max_size: "500MB"
      compression: "snappy"
  
  # 分析类型配置
  analysis_types:
    - name: "descriptive"
      include_statistics: true
      include_distribution: true
      include_correlation: true
    - name: "predictive"
      algorithms: ["linear_regression", "random_forest", "xgboost"]
      cross_validation: true
      feature_selection: true
    - name: "clustering"
      algorithms: ["kmeans", "dbscan", "hierarchical"]
      optimal_clusters: "auto"
    - name: "time_series"
      seasonality: "auto"
      trend: "auto"
      forecasting_periods: 30
  
  # 可视化配置
  visualization:
    library: "matplotlib"             # 可视化库
    style: "seaborn"                  # 样式
    figure_size: [12, 8]              # 图形大小
    dpi: 300                          # 分辨率
    formats: ["png", "svg", "pdf"]    # 输出格式
    color_palette: "viridis"          # 颜色方案
  
  # 统计配置
  statistics:
    confidence_level: 0.95            # 置信水平
    significance_level: 0.05          # 显著性水平
    missing_data_threshold: 0.1       # 缺失数据阈值
    outlier_detection: "iqr"          # 异常值检测方法

# 输出控制配置
output_control:
  quality_threshold: 0.8
  validation_rules:
    - name: "data_quality_check"
      enabled: true
    - name: "statistical_validity_check"
      enabled: true
    - name: "visualization_check"
      enabled: true
    - name: "interpretation_check"
      enabled: true
```

## LLM提供商配置

### OpenAI配置

```yaml
# configs/llm/openai.yaml

provider: "openai"
name: "OpenAI"
description: "OpenAI GPT模型配置"

# API配置
api:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  organization: "${OPENAI_ORG_ID}"     # 可选
  timeout: 60
  max_retries: 3
  retry_delay: 1

# 支持的模型
models:
  gpt-4:
    max_tokens: 8192
    context_window: 8192
    cost_per_1k_tokens:
      input: 0.03
      output: 0.06
    capabilities: ["text", "code", "reasoning"]
  
  gpt-4-turbo:
    max_tokens: 4096
    context_window: 128000
    cost_per_1k_tokens:
      input: 0.01
      output: 0.03
    capabilities: ["text", "code", "reasoning", "vision"]
  
  gpt-3.5-turbo:
    max_tokens: 4096
    context_window: 16385
    cost_per_1k_tokens:
      input: 0.0015
      output: 0.002
    capabilities: ["text", "code"]

# 默认参数
default_parameters:
  temperature: 0.7
  top_p: 1.0
  frequency_penalty: 0
  presence_penalty: 0
  stop: null

# 速率限制
rate_limits:
  requests_per_minute: 3500
  tokens_per_minute: 90000
  requests_per_day: 10000

# 错误处理
error_handling:
  retry_on_errors: ["rate_limit", "server_error", "timeout"]
  backoff_strategy: "exponential"
  max_backoff_time: 60
```

### Anthropic配置

```yaml
# configs/llm/anthropic.yaml

provider: "anthropic"
name: "Anthropic"
description: "Anthropic Claude模型配置"

# API配置
api:
  base_url: "https://api.anthropic.com"
  api_key: "${ANTHROPIC_API_KEY}"
  timeout: 60
  max_retries: 3
  retry_delay: 2

# 支持的模型
models:
  claude-3-opus:
    max_tokens: 4096
    context_window: 200000
    cost_per_1k_tokens:
      input: 0.015
      output: 0.075
    capabilities: ["text", "code", "reasoning", "vision"]
  
  claude-3-sonnet:
    max_tokens: 4096
    context_window: 200000
    cost_per_1k_tokens:
      input: 0.003
      output: 0.015
    capabilities: ["text", "code", "reasoning", "vision"]
  
  claude-3-haiku:
    max_tokens: 4096
    context_window: 200000
    cost_per_1k_tokens:
      input: 0.00025
      output: 0.00125
    capabilities: ["text", "code"]

# 默认参数
default_parameters:
  temperature: 0.7
  top_p: 1.0
  top_k: 40
  stop_sequences: []

# 速率限制
rate_limits:
  requests_per_minute: 1000
  tokens_per_minute: 40000
  requests_per_day: 5000
```

### 本地模型配置

```yaml
# configs/llm/local.yaml

provider: "local"
name: "Local Models"
description: "本地部署模型配置"

# Ollama配置
ollama:
  base_url: "http://localhost:11434"
  timeout: 120
  
  models:
    llama2:
      model_name: "llama2:7b"
      context_window: 4096
      capabilities: ["text", "code"]
    
    codellama:
      model_name: "codellama:7b"
      context_window: 4096
      capabilities: ["code"]
    
    mistral:
      model_name: "mistral:7b"
      context_window: 8192
      capabilities: ["text", "reasoning"]

# LM Studio配置
lm_studio:
  base_url: "http://localhost:1234/v1"
  timeout: 120
  
  models:
    local_model:
      model_name: "local-model"
      context_window: 4096
      capabilities: ["text"]

# vLLM配置
vllm:
  base_url: "http://localhost:8000/v1"
  timeout: 120
  
  models:
    custom_model:
      model_name: "custom-model"
      context_window: 8192
      capabilities: ["text", "code"]
```

## 输出控制配置

### 验证规则配置

```yaml
# configs/output_control/validation_rules.yaml

validation_rules:
  # 长度检查
  length_check:
    enabled: true
    min_length: 10
    max_length: 10000
    min_words: 5
    max_words: 2000
    severity: "error"
  
  # 格式检查
  format_check:
    enabled: true
    allowed_formats: ["markdown", "html", "plain_text"]
    require_structure: true
    severity: "warning"
  
  # 内容检查
  content_check:
    enabled: true
    check_grammar: true
    check_spelling: true
    check_coherence: true
    severity: "warning"
  
  # 安全检查
  security_check:
    enabled: true
    block_personal_info: true
    block_harmful_content: true
    block_copyrighted_content: false
    severity: "error"
  
  # 质量检查
  quality_check:
    enabled: true
    min_quality_score: 0.7
    check_relevance: true
    check_accuracy: true
    severity: "warning"

# 质量评估配置
quality_assessment:
  # 评估维度
  dimensions:
    relevance:
      weight: 0.3
      description: "内容与任务的相关性"
    accuracy:
      weight: 0.25
      description: "内容的准确性"
    completeness:
      weight: 0.2
      description: "内容的完整性"
    clarity:
      weight: 0.15
      description: "内容的清晰度"
    creativity:
      weight: 0.1
      description: "内容的创新性"
  
  # 评估方法
  methods:
    - name: "llm_based"
      enabled: true
      model: "gpt-4"
      prompt_template: "quality_assessment_prompt.txt"
    - name: "rule_based"
      enabled: true
      rules: ["length", "structure", "grammar"]
    - name: "similarity_based"
      enabled: false
      reference_corpus: "quality_examples.json"

# 重试策略配置
retry_strategy:
  max_retries: 3
  retry_delay: 2
  backoff_multiplier: 2
  retry_on_failures:
    - "quality_too_low"
    - "validation_failed"
    - "format_error"
  
  improvement_prompts:
    quality_too_low: "请提高输出质量，确保内容更加准确和相关。"
    validation_failed: "请修正输出中的错误，确保符合要求。"
    format_error: "请调整输出格式，确保符合指定的格式要求。"
```

## 工具配置

### 工具注册配置

```yaml
# configs/tools/tool_registry.yaml

tools:
  # 分析工具
  analysis:
    - name: "data_analyzer"
      class: "src.tools.analysis.data_analyzer.DataAnalyzer"
      enabled: true
      config:
        max_data_size: "100MB"
        supported_formats: ["csv", "json", "excel"]
    
    - name: "text_analyzer"
      class: "src.tools.analysis.text_analyzer.TextAnalyzer"
      enabled: true
      config:
        languages: ["en", "zh", "es"]
        sentiment_analysis: true
  
  # 开发工具
  development:
    - name: "code_executor"
      class: "src.tools.development.code_executor.CodeExecutor"
      enabled: true
      config:
        supported_languages: ["python", "javascript"]
        timeout: 30
        sandbox: true
    
    - name: "test_generator"
      class: "src.tools.development.test_generator.TestGenerator"
      enabled: true
      config:
        frameworks: ["pytest", "jest", "junit"]
        coverage_threshold: 0.8
  
  # 外部工具
  external:
    - name: "web_searcher"
      class: "src.tools.external.web_searcher.WebSearcher"
      enabled: true
      config:
        search_engine: "google"
        max_results: 10
        timeout: 15
    
    - name: "api_caller"
      class: "src.tools.external.api_caller.APICaller"
      enabled: true
      config:
        timeout: 30
        max_retries: 3
        rate_limit: 100
  
  # 操作工具
  operations:
    - name: "file_manager"
      class: "src.tools.operations.file_manager.FileManager"
      enabled: true
      config:
        allowed_extensions: [".txt", ".md", ".json", ".csv"]
        max_file_size: "10MB"
        base_directory: "./workspace"
    
    - name: "email_sender"
      class: "src.tools.operations.email_sender.EmailSender"
      enabled: false
      config:
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        use_tls: true

# 工具安全配置
security:
  sandbox_mode: true
  allowed_operations:
    - "read_file"
    - "write_file"
    - "execute_code"
    - "web_request"
  
  blocked_operations:
    - "system_command"
    - "network_scan"
    - "file_delete"
  
  resource_limits:
    max_memory: "512MB"
    max_cpu_time: 30
    max_network_requests: 100
```

## 日志配置

### 详细日志配置

```yaml
# configs/logging/logging.yaml

version: 1
disable_existing_loggers: false

# 格式化器
formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  detailed:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
  
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
    datefmt: "%Y-%m-%d %H:%M:%S"

# 处理器
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/agent_workflow.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    encoding: utf-8
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log
    maxBytes: 10485760  # 10MB
    backupCount: 3
    encoding: utf-8
  
  json_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: logs/agent_workflow.json
    maxBytes: 10485760  # 10MB
    backupCount: 5
    encoding: utf-8

# 记录器
loggers:
  src.agents:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  src.core:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  src.tools:
    level: INFO
    handlers: [console, file]
    propagate: false
  
  src.workflows:
    level: INFO
    handlers: [console, file]
    propagate: false

# 根记录器
root:
  level: INFO
  handlers: [console, file, error_file]

# 特殊配置
special_loggers:
  performance:
    logger_name: "performance"
    level: INFO
    handlers: [json_file]
    metrics:
      - "execution_time"
      - "memory_usage"
      - "token_usage"
      - "api_calls"
  
  audit:
    logger_name: "audit"
    level: INFO
    handlers: [json_file]
    events:
      - "task_started"
      - "task_completed"
      - "task_failed"
      - "config_changed"
      - "user_action"
```

## 性能配置

### 性能优化配置

```yaml
# configs/performance/optimization.yaml

# 缓存配置
caching:
  enabled: true
  backend: "redis"                    # redis, memory, file
  
  redis:
    host: "localhost"
    port: 6379
    db: 0
    password: null
    connection_pool_size: 10
  
  memory:
    max_size: "256MB"
    ttl: 3600
  
  file:
    cache_dir: "./cache"
    max_size: "1GB"
    ttl: 7200
  
  # 缓存策略
  strategies:
    llm_responses:
      enabled: true
      ttl: 3600
      key_pattern: "llm:{provider}:{model}:{hash}"
    
    tool_results:
      enabled: true
      ttl: 1800
      key_pattern: "tool:{tool_name}:{hash}"
    
    validation_results:
      enabled: true
      ttl: 900
      key_pattern: "validation:{rule}:{hash}"

# 并发配置
concurrency:
  max_concurrent_tasks: 10
  max_concurrent_llm_calls: 5
  max_concurrent_tool_calls: 8
  
  # 线程池配置
  thread_pools:
    io_bound:
      max_workers: 20
      thread_name_prefix: "io-worker"
    
    cpu_bound:
      max_workers: 4
      thread_name_prefix: "cpu-worker"
  
  # 异步配置
  async_settings:
    event_loop_policy: "asyncio.DefaultEventLoopPolicy"
    max_async_tasks: 100
    task_timeout: 300

# 内存管理
memory:
  max_memory_usage: "2GB"
  memory_check_interval: 60
  gc_threshold: 0.8
  
  # 内存监控
  monitoring:
    enabled: true
    log_interval: 300
    alert_threshold: 0.9

# 连接池配置
connection_pools:
  http:
    max_connections: 100
    max_connections_per_host: 20
    timeout: 30
    keepalive_timeout: 30
  
  database:
    max_connections: 20
    min_connections: 5
    timeout: 30
    retry_attempts: 3

# 批处理配置
batch_processing:
  enabled: true
  batch_size: 10
  batch_timeout: 5
  max_batch_wait_time: 30
  
  # 批处理策略
  strategies:
    llm_calls:
      enabled: true
      batch_size: 5
      timeout: 10
    
    tool_calls:
      enabled: true
      batch_size: 8
      timeout: 15
```

## 安全配置

### 安全策略配置

```yaml
# configs/security/security.yaml

# 认证配置
authentication:
  enabled: true
  method: "api_key"                   # api_key, oauth, jwt
  
  api_key:
    header_name: "X-API-Key"
    required: true
    validation_endpoint: null
  
  oauth:
    provider: "google"
    client_id: "${OAUTH_CLIENT_ID}"
    client_secret: "${OAUTH_CLIENT_SECRET}"
    scopes: ["openid", "email", "profile"]
  
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expiration_time: 3600

# 授权配置
authorization:
  enabled: true
  model: "rbac"                       # rbac, abac
  
  roles:
    admin:
      permissions: ["*"]
    user:
      permissions: ["read", "execute"]
    guest:
      permissions: ["read"]
  
  resources:
    agents:
      permissions: ["create", "read", "update", "delete", "execute"]
    tools:
      permissions: ["read", "execute"]
    configs:
      permissions: ["read", "update"]

# 输入验证
input_validation:
  enabled: true
  
  # 输入清理
  sanitization:
    remove_html: true
    remove_scripts: true
    escape_special_chars: true
    max_input_length: 10000
  
  # 内容过滤
  content_filtering:
    block_personal_info: true
    block_harmful_content: true
    block_spam: true
    
    patterns:
      email: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      phone: "\\b\\d{3}-\\d{3}-\\d{4}\\b"
      ssn: "\\b\\d{3}-\\d{2}-\\d{4}\\b"

# 速率限制
rate_limiting:
  enabled: true
  
  global:
    requests_per_minute: 1000
    requests_per_hour: 10000
    requests_per_day: 100000
  
  per_user:
    requests_per_minute: 100
    requests_per_hour: 1000
    requests_per_day: 10000
  
  per_endpoint:
    "/api/agents/execute":
      requests_per_minute: 50
    "/api/tools/execute":
      requests_per_minute: 100

# 数据保护
data_protection:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation_interval: 2592000  # 30 days
  
  data_retention:
    logs: 90                        # days
    cache: 7                        # days
    user_data: 365                  # days
  
  privacy:
    anonymize_logs: true
    mask_sensitive_data: true
    gdpr_compliance: true

# 监控和审计
monitoring:
  enabled: true
  
  security_events:
    - "authentication_failure"
    - "authorization_failure"
    - "rate_limit_exceeded"
    - "suspicious_activity"
    - "data_breach_attempt"
  
  alerting:
    enabled: true
    channels: ["email", "slack"]
    thresholds:
      failed_logins: 5
      rate_limit_violations: 10
      suspicious_patterns: 3
```

## 环境变量

### 环境变量配置

```bash
# .env - 环境变量配置文件

# LLM API密钥
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/agent_workflow
REDIS_URL=redis://localhost:6379/0

# 安全配置
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# 外部服务
WEB_SEARCH_API_KEY=your_search_api_key_here
EMAIL_SMTP_PASSWORD=your_email_password_here

# 应用配置
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# 性能配置
MAX_WORKERS=10
CACHE_TTL=3600
REQUEST_TIMEOUT=30

# 监控配置
MONITORING_ENABLED=true
METRICS_ENDPOINT=http://localhost:9090
TRACING_ENDPOINT=http://localhost:14268
```

### 环境特定配置

```yaml
# configs/environments/development.yaml

environment: "development"

# 开发环境特定配置
development:
  debug: true
  hot_reload: true
  detailed_logging: true
  
  # 宽松的限制
  rate_limits:
    requests_per_minute: 10000
  
  # 快速缓存过期
  cache_ttl: 300
  
  # 详细错误信息
  error_details: true

# 覆盖生产配置
overrides:
  logging:
    level: "DEBUG"
  
  security:
    authentication:
      enabled: false
  
  performance:
    caching:
      enabled: false
```

```yaml
# configs/environments/production.yaml

environment: "production"

# 生产环境特定配置
production:
  debug: false
  hot_reload: false
  detailed_logging: false
  
  # 严格的限制
  rate_limits:
    requests_per_minute: 1000
  
  # 长缓存过期
  cache_ttl: 3600
  
  # 简化错误信息
  error_details: false

# 覆盖开发配置
overrides:
  logging:
    level: "INFO"
  
  security:
    authentication:
      enabled: true
    authorization:
      enabled: true
  
  performance:
    caching:
      enabled: true
    concurrency:
      max_concurrent_tasks: 20
```

## 配置最佳实践

### 1. 配置文件组织

- **分层配置**: 使用基础配置 + 环境特定配置
- **模块化**: 按功能模块分离配置文件
- **版本控制**: 配置文件纳入版本控制，敏感信息使用环境变量
- **文档化**: 为每个配置项添加注释说明

### 2. 安全考虑

- **敏感信息**: 使用环境变量存储API密钥等敏感信息
- **权限控制**: 限制配置文件的读写权限
- **加密存储**: 对敏感配置进行加密存储
- **定期轮换**: 定期更换API密钥和加密密钥

### 3. 性能优化

- **缓存策略**: 合理设置缓存TTL和大小限制
- **并发控制**: 根据系统资源调整并发参数
- **资源限制**: 设置内存和CPU使用限制
- **监控指标**: 配置性能监控和告警

### 4. 配置验证

```python
# 配置验证示例
from src.core.config.config_validator import ConfigValidator

def validate_config(config_path: str) -> bool:
    """验证配置文件"""
    validator = ConfigValidator()
    
    try:
        # 加载配置
        config = validator.load_config(config_path)
        
        # 验证必需字段
        validator.validate_required_fields(config)
        
        # 验证数据类型
        validator.validate_data_types(config)
        
        # 验证值范围
        validator.validate_value_ranges(config)
        
        # 验证依赖关系
        validator.validate_dependencies(config)
        
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False
```

### 5. 配置热重载

```python
# 配置热重载示例
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.src_path.endswith('.yaml'):
            print(f"配置文件已修改: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_config())

# 启用配置监控
def enable_config_monitoring(config_dir: str, config_manager):
    event_handler = ConfigReloadHandler(config_manager)
    observer = Observer()
    observer.schedule(event_handler, config_dir, recursive=True)
    observer.start()
    return observer
```

---

通过遵循本配置指南，您可以有效地配置和管理AI Agents Workflow框架，确保系统的稳定性、安全性和性能。