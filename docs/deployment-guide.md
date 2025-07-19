# 部署指南

本指南详细介绍如何在不同环境中部署AI Agents Workflow框架，包括本地开发、测试环境和生产环境的部署方案。

## 目录

- [环境要求](#环境要求)
- [本地开发部署](#本地开发部署)
- [Docker部署](#docker部署)
- [云平台部署](#云平台部署)
- [生产环境部署](#生产环境部署)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

## 环境要求

### 系统要求

**最低要求：**
- CPU: 2核心
- 内存: 4GB RAM
- 存储: 10GB 可用空间
- 操作系统: Windows 10+, macOS 10.15+, Ubuntu 18.04+

**推荐配置：**
- CPU: 4核心或更多
- 内存: 8GB RAM 或更多
- 存储: 50GB SSD
- 网络: 稳定的互联网连接

### 软件依赖

**必需软件：**
- Python 3.9+
- pip 或 uv (包管理器)
- Git

**可选软件：**
- Docker & Docker Compose
- Redis (缓存)
- PostgreSQL (数据存储)
- Nginx (反向代理)

### Python依赖

```bash
# 核心依赖
python>=3.9
aiohttp>=3.8.0
pydantic>=2.0.0
pyyaml>=6.0
click>=8.0.0

# LLM客户端
openai>=1.0.0
anthropic>=0.3.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# 缓存和存储
redis>=4.5.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0

# 监控和日志
prometheus-client>=0.16.0
structlog>=23.1.0

# 开发工具
pytest>=7.0.0
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0
```

## 本地开发部署

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-org/ai-agents-workflow.git
cd ai-agents-workflow

# 2. 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -e .

# 或使用uv (推荐)
uv sync

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加必要的API密钥

# 5. 运行示例
python -m AgentWorker.CodeAssistant.code_assistant_worker
```

### 开发环境配置

```yaml
# configs/environments/development.yaml

environment: "development"

# 开发环境配置
development:
  debug: true
  hot_reload: true
  auto_restart: true
  
  # 数据库配置
  database:
    url: "sqlite:///./dev.db"
    echo: true
    create_tables: true
  
  # 缓存配置
  cache:
    backend: "memory"
    ttl: 300
  
  # 日志配置
  logging:
    level: "DEBUG"
    console: true
    file: false
  
  # API配置
  api:
    host: "localhost"
    port: 8000
    reload: true
    workers: 1
```

### IDE配置

**VSCode配置 (.vscode/settings.json):**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.analysis.extraPaths": ["./src"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true
    }
}
```

**PyCharm配置:**
1. 设置Python解释器为项目虚拟环境
2. 标记 `src` 目录为源代码根目录
3. 配置代码格式化工具 (Black, isort)
4. 启用类型检查 (mypy)
5. 配置测试运行器 (pytest)

## Docker部署

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装uv
RUN pip install uv

# 安装Python依赖
RUN uv sync --frozen

# 复制源代码
COPY src/ ./src/
COPY AgentWorker/ ./AgentWorker/
COPY configs/ ./configs/

# 创建日志目录
RUN mkdir -p logs

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动命令
CMD ["python", "-m", "src.api.main"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 主应用
  agent-workflow:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agent_workflow
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./configs:/app/configs
    restart: unless-stopped
    networks:
      - agent-network
  
  # PostgreSQL数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: agent_workflow
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - agent-network
  
  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - agent-network
  
  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - agent-workflow
    restart: unless-stopped
    networks:
      - agent-network
  
  # 监控服务
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    networks:
      - agent-network
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped
    networks:
      - agent-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  agent-network:
    driver: bridge
```

### 部署命令

```bash
# 构建和启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f agent-workflow

# 停止服务
docker-compose down

# 重新构建
docker-compose up -d --build

# 扩展服务
docker-compose up -d --scale agent-workflow=3
```

## 云平台部署

### AWS部署

#### ECS部署

```yaml
# aws/ecs-task-definition.json
{
  "family": "agent-workflow",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "agent-workflow",
      "image": "your-account.dkr.ecr.region.amazonaws.com/agent-workflow:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/agent-workflow",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### 部署脚本

```bash
#!/bin/bash
# aws/deploy.sh

set -e

# 配置变量
REGION="us-west-2"
ACCOUNT_ID="your-account-id"
REPO_NAME="agent-workflow"
CLUSTER_NAME="agent-workflow-cluster"
SERVICE_NAME="agent-workflow-service"

# 构建和推送Docker镜像
echo "Building Docker image..."
docker build -t $REPO_NAME .

# 登录ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 标记和推送镜像
echo "Pushing image to ECR..."
docker tag $REPO_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# 更新ECS服务
echo "Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force-new-deployment \
    --region $REGION

echo "Deployment completed!"
```

### Google Cloud Platform部署

#### Cloud Run部署

```yaml
# gcp/cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agent-workflow
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/your-project/agent-workflow:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-api-key
              key: key
        resources:
          limits:
            cpu: "1000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### 部署脚本

```bash
#!/bin/bash
# gcp/deploy.sh

set -e

# 配置变量
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="agent-workflow"

# 构建和推送镜像
echo "Building and pushing image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# 部署到Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars ENVIRONMENT=production

echo "Deployment completed!"
```

### Azure部署

#### Container Instances部署

```yaml
# azure/container-group.yaml
apiVersion: 2021-03-01
location: eastus
name: agent-workflow
properties:
  containers:
  - name: agent-workflow
    properties:
      image: your-registry.azurecr.io/agent-workflow:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: OPENAI_API_KEY
        secureValue: your-api-key
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
type: Microsoft.ContainerInstance/containerGroups
```

## 生产环境部署

### 生产环境配置

```yaml
# configs/environments/production.yaml

environment: "production"

# 生产环境配置
production:
  debug: false
  
  # 数据库配置
  database:
    url: "${DATABASE_URL}"
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    pool_recycle: 3600
  
  # 缓存配置
  cache:
    backend: "redis"
    url: "${REDIS_URL}"
    pool_size: 10
    ttl: 3600
  
  # 日志配置
  logging:
    level: "INFO"
    console: false
    file: true
    structured: true
    format: "json"
  
  # API配置
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    worker_class: "uvicorn.workers.UvicornWorker"
    max_requests: 1000
    max_requests_jitter: 100
  
  # 安全配置
  security:
    authentication: true
    authorization: true
    rate_limiting: true
    input_validation: true
  
  # 监控配置
  monitoring:
    metrics: true
    tracing: true
    health_checks: true
    alerts: true
```

### 负载均衡配置

```nginx
# nginx/nginx.conf
upstream agent_workflow {
    least_conn;
    server agent-workflow-1:8000 max_fails=3 fail_timeout=30s;
    server agent-workflow-2:8000 max_fails=3 fail_timeout=30s;
    server agent-workflow-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # 重定向到HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL配置
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # 代理配置
    location / {
        proxy_pass http://agent_workflow;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # 缓冲配置
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://agent_workflow/health;
        access_log off;
    }
    
    # 静态文件
    location /static/ {
        alias /var/www/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 数据库迁移

```python
# scripts/migrate.py
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from src.core.database.models import Base
from src.core.config.config_manager import ConfigManager

async def migrate_database():
    """执行数据库迁移"""
    config = ConfigManager().load_config('production.yaml')
    database_url = config.database.url
    
    # 创建异步引擎
    engine = create_async_engine(database_url)
    
    try:
        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("数据库迁移完成")
        
    except Exception as e:
        print(f"数据库迁移失败: {e}")
        raise
    
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate_database())
```

## 监控和维护

### 健康检查

```python
# src/api/health.py
from fastapi import APIRouter, HTTPException
from src.core.health.health_checker import HealthChecker

router = APIRouter()
health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """基础健康检查"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    checks = await health_checker.run_all_checks()
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "checks": checks
    }

@router.get("/health/ready")
async def readiness_check():
    """就绪检查"""
    ready = await health_checker.check_readiness()
    
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.utcnow()}
```

### 监控指标

```python
# src/core/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
request_count = Counter(
    'agent_requests_total',
    'Total number of requests',
    ['agent_type', 'status']
)

request_duration = Histogram(
    'agent_request_duration_seconds',
    'Request duration in seconds',
    ['agent_type']
)

active_tasks = Gauge(
    'agent_active_tasks',
    'Number of active tasks',
    ['agent_type']
)

llm_api_calls = Counter(
    'llm_api_calls_total',
    'Total number of LLM API calls',
    ['provider', 'model', 'status']
)

class MetricsCollector:
    def __init__(self):
        self.start_metrics_server()
    
    def start_metrics_server(self, port=9090):
        """启动指标服务器"""
        start_http_server(port)
    
    def record_request(self, agent_type: str, status: str, duration: float):
        """记录请求指标"""
        request_count.labels(agent_type=agent_type, status=status).inc()
        request_duration.labels(agent_type=agent_type).observe(duration)
    
    def update_active_tasks(self, agent_type: str, count: int):
        """更新活跃任务数"""
        active_tasks.labels(agent_type=agent_type).set(count)
    
    def record_llm_call(self, provider: str, model: str, status: str):
        """记录LLM调用"""
        llm_api_calls.labels(provider=provider, model=model, status=status).inc()
```

### 日志聚合

```yaml
# monitoring/fluentd.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<filter agent.**>
  @type parser
  key_name log
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<match agent.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name agent-logs
  type_name _doc
  
  <buffer>
    @type file
    path /var/log/fluentd-buffers/agent.buffer
    flush_mode interval
    flush_interval 10s
  </buffer>
</match>
```

### 告警配置

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@your-domain.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@your-domain.com'
    subject: 'Agent Workflow Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Agent Workflow Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## 故障排除

### 常见问题

#### 1. 内存不足

**症状：**
- 应用崩溃或重启
- 响应时间变慢
- OOM错误

**解决方案：**
```bash
# 检查内存使用
docker stats

# 增加内存限制
docker run -m 4g your-image

# 优化代码
# - 使用生成器而不是列表
# - 及时释放大对象
# - 启用垃圾回收
```

#### 2. API调用失败

**症状：**
- LLM API调用超时
- 认证错误
- 速率限制

**解决方案：**
```python
# 检查API密钥
import os
print(os.getenv('OPENAI_API_KEY'))

# 增加超时时间
client = OpenAI(timeout=60)

# 实现重试机制
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_llm_api():
    # API调用逻辑
    pass
```

#### 3. 数据库连接问题

**症状：**
- 连接超时
- 连接池耗尽
- 死锁

**解决方案：**
```python
# 检查连接池配置
engine = create_engine(
    database_url,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600
)

# 使用连接上下文管理器
async with engine.begin() as conn:
    result = await conn.execute(query)
```

### 调试工具

```python
# scripts/debug.py
import asyncio
import logging
from src.core.config.config_manager import ConfigManager
from src.agents.factory.agent_factory import AgentFactory

async def debug_agent(agent_type: str):
    """调试代理"""
    # 启用详细日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 加载配置
    config = ConfigManager().load_config(f'{agent_type}_config.yaml')
    
    # 创建代理
    agent = AgentFactory.create_agent(agent_type, config)
    
    # 测试任务
    task = TaskDefinition(
        task_type='test',
        prompt='测试提示',
        parameters={'test': True}
    )
    
    try:
        result = await agent.process_task(task)
        print(f"成功: {result}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_agent('code_assistant'))
```

## 性能优化

### 缓存策略

```python
# src/core/cache/cache_manager.py
import redis
from typing import Any, Optional
import json
import hashlib

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def _generate_key(self, prefix: str, data: dict) -> str:
        """生成缓存键"""
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, prefix: str, data: dict) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(prefix, data)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    async def set(self, prefix: str, data: dict, value: Any, ttl: int = 3600):
        """设置缓存"""
        key = self._generate_key(prefix, data)
        self.redis.setex(key, ttl, json.dumps(value))
    
    async def invalidate_pattern(self, pattern: str):
        """按模式清除缓存"""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

### 连接池优化

```python
# src/core/http/client_pool.py
import aiohttp
from typing import Optional

class HTTPClientPool:
    def __init__(self, max_connections: int = 100):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout
            )
        return self.session
    
    async def close(self):
        """关闭连接池"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.connector.close()
```

### 批处理优化

```python
# src/core/batch/batch_processor.py
import asyncio
from typing import List, Callable, Any
from collections import defaultdict

class BatchProcessor:
    def __init__(self, batch_size: int = 10, timeout: float = 5.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.batches = defaultdict(list)
        self.processors = {}
    
    async def add_task(self, batch_key: str, task: Any, processor: Callable):
        """添加任务到批处理队列"""
        self.batches[batch_key].append(task)
        self.processors[batch_key] = processor
        
        # 检查是否需要处理批次
        if len(self.batches[batch_key]) >= self.batch_size:
            await self._process_batch(batch_key)
    
    async def _process_batch(self, batch_key: str):
        """处理批次"""
        if batch_key not in self.batches or not self.batches[batch_key]:
            return
        
        batch = self.batches[batch_key][:self.batch_size]
        self.batches[batch_key] = self.batches[batch_key][self.batch_size:]
        
        processor = self.processors[batch_key]
        await processor(batch)
    
    async def flush_all(self):
        """处理所有待处理的批次"""
        tasks = []
        for batch_key in list(self.batches.keys()):
            if self.batches[batch_key]:
                tasks.append(self._process_batch(batch_key))
        
        if tasks:
            await asyncio.gather(*tasks)
```

---

通过遵循本部署指南，您可以在各种环境中成功部署和运行AI Agents Workflow框架，确保系统的稳定性、可扩展性和高可用性。