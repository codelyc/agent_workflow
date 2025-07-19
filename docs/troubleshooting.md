# 故障排除指南

本指南提供了AI Agents Workflow框架常见问题的诊断和解决方案，帮助您快速识别和修复问题。

## 目录

- [快速诊断](#快速诊断)
- [安装和配置问题](#安装和配置问题)
- [运行时错误](#运行时错误)
- [性能问题](#性能问题)
- [API和网络问题](#api和网络问题)
- [数据库问题](#数据库问题)
- [内存和资源问题](#内存和资源问题)
- [日志分析](#日志分析)
- [调试工具](#调试工具)
- [常见错误代码](#常见错误代码)

## 快速诊断

### 系统健康检查

```bash
# 检查Python环境
python --version
pip list | grep -E "(aiohttp|pydantic|openai)"

# 检查项目结构
ls -la src/
ls -la AgentWorker/

# 检查配置文件
cat configs/default_config.yaml

# 测试基本导入
python -c "from src.core.config.config_manager import ConfigManager; print('导入成功')"
```

### 快速测试脚本

```python
# scripts/quick_test.py
import sys
import os
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

async def quick_test():
    """快速测试系统组件"""
    print("🔍 开始快速诊断...")
    
    # 测试1: 导入核心模块
    try:
        from src.core.config.config_manager import ConfigManager
        from src.core.output_control import OutputController
        print("✅ 核心模块导入成功")
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        return False
    
    # 测试2: 配置管理器
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config("default_config.yaml")
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 测试3: 输出控制器
    try:
        output_controller = OutputController()
        await output_controller.display_message("测试消息", "info")
        print("✅ 输出控制器工作正常")
    except Exception as e:
        print(f"❌ 输出控制器失败: {e}")
        return False
    
    # 测试4: 环境变量
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ API密钥已配置")
    else:
        print("⚠️ 未找到OPENAI_API_KEY环境变量")
    
    print("🎉 快速诊断完成")
    return True

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
```

## 安装和配置问题

### 问题1: 模块导入失败

**错误信息：**
```
ModuleNotFoundError: No module named 'src.core.config'
```

**原因分析：**
- Python路径配置不正确
- 项目结构问题
- 虚拟环境未激活

**解决方案：**

```bash
# 方案1: 检查并修复Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"

# 方案2: 安装为可编辑包
pip install -e .

# 方案3: 创建.pth文件
echo "$(pwd)/src" > $(python -c "import site; print(site.getsitepackages()[0])")/agent_workflow.pth

# 方案4: 检查虚拟环境
which python
pip list
```

**IDE配置修复：**

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.analysis.extraPaths": ["./src", "./AgentWorker"],
    "python.analysis.autoSearchPaths": true,
    "python.analysis.autoImportCompletions": true
}
```

### 问题2: 依赖版本冲突

**错误信息：**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**解决方案：**

```bash
# 清理环境
pip freeze > requirements_backup.txt
pip uninstall -r requirements_backup.txt -y

# 重新安装
pip install --upgrade pip
pip install -r requirements.txt

# 或使用uv (推荐)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 问题3: 配置文件找不到

**错误信息：**
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/default_config.yaml'
```

**解决方案：**

```python
# 检查配置文件路径
import os
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent
config_path = project_root / "configs" / "default_config.yaml"

print(f"配置文件路径: {config_path}")
print(f"文件存在: {config_path.exists()}")

# 如果文件不存在，创建默认配置
if not config_path.exists():
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write('''
# 默认配置
version: "1.0"

# 代理配置
agents:
  code_assistant:
    enabled: true
    model: "gpt-3.5-turbo"
    max_tokens: 2000

# LLM配置
llm:
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"

# 输出配置
output:
  console:
    enabled: true
    level: "info"
  file:
    enabled: false
        ''')
```

## 运行时错误

### 问题1: 异步函数调用错误

**错误信息：**
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**原因分析：**
- 在已有事件循环中调用`asyncio.run()`
- Jupyter notebook环境问题

**解决方案：**

```python
# 错误的调用方式
# asyncio.run(some_async_function())

# 正确的调用方式
import asyncio

# 方案1: 检查是否有运行中的事件循环
try:
    loop = asyncio.get_running_loop()
    # 如果有运行中的循环，直接await
    result = await some_async_function()
except RuntimeError:
    # 如果没有运行中的循环，使用asyncio.run
    result = asyncio.run(some_async_function())

# 方案2: 使用nest_asyncio (适用于Jupyter)
import nest_asyncio
nest_asyncio.apply()
result = asyncio.run(some_async_function())

# 方案3: 创建任务
loop = asyncio.get_event_loop()
task = loop.create_task(some_async_function())
result = await task
```

### 问题2: 任务超时

**错误信息：**
```
asyncio.TimeoutError: Task timed out after 30 seconds
```

**解决方案：**

```python
# 增加超时时间
import asyncio

async def process_with_timeout(task, timeout=60):
    try:
        result = await asyncio.wait_for(task, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        print(f"任务超时 ({timeout}秒)")
        # 实现重试逻辑
        return await retry_task(task)

async def retry_task(task, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(task, timeout=120)
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 指数退避
```

### 问题3: 内存泄漏

**症状：**
- 内存使用持续增长
- 应用变慢或崩溃

**诊断工具：**

```python
# memory_profiler.py
import psutil
import gc
import tracemalloc
from typing import Dict, Any

class MemoryProfiler:
    def __init__(self):
        self.start_memory = None
        tracemalloc.start()
    
    def start_profiling(self):
        """开始内存分析"""
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"开始内存使用: {self.start_memory:.2f} MB")
    
    def check_memory(self, label: str = ""):
        """检查当前内存使用"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        if self.start_memory:
            diff = current_memory - self.start_memory
            print(f"{label} 内存使用: {current_memory:.2f} MB (+{diff:.2f} MB)")
        else:
            print(f"{label} 内存使用: {current_memory:.2f} MB")
    
    def get_top_memory_objects(self, limit=10):
        """获取占用内存最多的对象"""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print(f"Top {limit} memory consumers:")
        for index, stat in enumerate(top_stats[:limit], 1):
            print(f"{index}. {stat}")
    
    def force_gc(self):
        """强制垃圾回收"""
        before = psutil.Process().memory_info().rss / 1024 / 1024
        collected = gc.collect()
        after = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"垃圾回收: 回收了 {collected} 个对象, 释放了 {before-after:.2f} MB")

# 使用示例
profiler = MemoryProfiler()
profiler.start_profiling()

# 在关键点检查内存
profiler.check_memory("处理任务后")
profiler.get_top_memory_objects()
profiler.force_gc()
```

## 性能问题

### 问题1: 响应时间过长

**诊断工具：**

```python
# performance_monitor.py
import time
import asyncio
from functools import wraps
from typing import Callable, Any

def performance_monitor(func: Callable) -> Callable:
    """性能监控装饰器"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            print(f"{func.__name__} 执行时间: {duration:.2f}秒")
            
            # 如果执行时间过长，记录警告
            if duration > 10:
                print(f"⚠️ {func.__name__} 执行时间过长: {duration:.2f}秒")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            print(f"{func.__name__} 执行时间: {duration:.2f}秒")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# 使用示例
@performance_monitor
async def slow_function():
    await asyncio.sleep(5)
    return "完成"
```

**优化策略：**

```python
# 并发处理
async def process_tasks_concurrently(tasks, max_concurrent=5):
    """并发处理任务"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await process_single_task(task)
    
    results = await asyncio.gather(*[
        process_with_semaphore(task) for task in tasks
    ])
    return results

# 缓存结果
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_data):
    """缓存计算结果"""
    # 耗时计算
    return result

# 批处理
async def batch_process(items, batch_size=10):
    """批量处理"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await process_batch(batch)
        await asyncio.sleep(0.1)  # 避免过载
```

### 问题2: CPU使用率过高

**诊断脚本：**

```python
# cpu_monitor.py
import psutil
import time
import threading
from collections import deque

class CPUMonitor:
    def __init__(self, window_size=60):
        self.cpu_history = deque(maxlen=window_size)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始CPU监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止CPU监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append(cpu_percent)
            
            if cpu_percent > 80:
                print(f"⚠️ CPU使用率过高: {cpu_percent}%")
                self._analyze_high_cpu()
    
    def _analyze_high_cpu(self):
        """分析高CPU使用率"""
        # 获取进程信息
        process = psutil.Process()
        threads = process.threads()
        
        print(f"当前进程线程数: {len(threads)}")
        print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # 检查子进程
        children = process.children(recursive=True)
        if children:
            print(f"子进程数: {len(children)}")
    
    def get_average_cpu(self):
        """获取平均CPU使用率"""
        if self.cpu_history:
            return sum(self.cpu_history) / len(self.cpu_history)
        return 0

# 使用示例
monitor = CPUMonitor()
monitor.start_monitoring()

# 运行你的代码
# ...

print(f"平均CPU使用率: {monitor.get_average_cpu():.2f}%")
monitor.stop_monitoring()
```

## API和网络问题

### 问题1: API调用失败

**错误信息：**
```
openai.error.RateLimitError: Rate limit exceeded
openai.error.AuthenticationError: Invalid API key
openai.error.APIConnectionError: Connection error
```

**解决方案：**

```python
# api_client_with_retry.py
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional

class RobustAPIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self.session
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """发起API请求"""
        session = await self.get_session()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with session.post(url, json=data) as response:
                if response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"速率限制，等待 {retry_after} 秒")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limit exceeded")
                
                if response.status == 401:  # Authentication error
                    raise ValueError("API密钥无效")
                
                response.raise_for_status()
                return await response.json()
        
        except aiohttp.ClientError as e:
            print(f"API请求失败: {e}")
            raise
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()

# 网络连接测试
async def test_network_connectivity():
    """测试网络连接"""
    test_urls = [
        "https://api.openai.com",
        "https://api.anthropic.com",
        "https://www.google.com"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in test_urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    print(f"✅ {url}: {response.status}")
            except Exception as e:
                print(f"❌ {url}: {e}")
```

### 问题2: 代理和防火墙问题

**解决方案：**

```python
# proxy_config.py
import os
import aiohttp

def setup_proxy_session():
    """配置代理会话"""
    proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
    
    if proxy_url:
        print(f"使用代理: {proxy_url}")
        connector = aiohttp.TCPConnector(
            limit=10,
            verify_ssl=False  # 如果代理有SSL问题
        )
        return aiohttp.ClientSession(
            connector=connector,
            trust_env=True  # 使用环境变量中的代理设置
        )
    else:
        return aiohttp.ClientSession()

# 环境变量设置
# export HTTP_PROXY=http://proxy.company.com:8080
# export HTTPS_PROXY=http://proxy.company.com:8080
# export NO_PROXY=localhost,127.0.0.1,.local
```

## 数据库问题

### 问题1: 连接池耗尽

**错误信息：**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached
```

**解决方案：**

```python
# database_config.py
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import QueuePool

def create_optimized_engine(database_url: str):
    """创建优化的数据库引擎"""
    return create_async_engine(
        database_url,
        # 连接池配置
        pool_size=20,          # 连接池大小
        max_overflow=30,       # 最大溢出连接数
        pool_timeout=30,       # 获取连接超时时间
        pool_recycle=3600,     # 连接回收时间
        pool_pre_ping=True,    # 连接前ping测试
        
        # 其他配置
        echo=False,            # 生产环境关闭SQL日志
        future=True,
        
        # 连接参数
        connect_args={
            "command_timeout": 60,
            "server_settings": {
                "application_name": "agent_workflow",
            }
        }
    )

# 连接管理器
class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_optimized_engine(database_url)
        self.active_connections = 0
    
    async def get_connection(self):
        """获取数据库连接"""
        try:
            self.active_connections += 1
            async with self.engine.begin() as conn:
                yield conn
        finally:
            self.active_connections -= 1
    
    async def check_pool_status(self):
        """检查连接池状态"""
        pool = self.engine.pool
        print(f"连接池状态:")
        print(f"  大小: {pool.size()}")
        print(f"  已检出: {pool.checkedout()}")
        print(f"  溢出: {pool.overflow()}")
        print(f"  无效: {pool.invalidated()}")
```

### 问题2: 死锁和长时间运行的查询

**诊断工具：**

```sql
-- PostgreSQL死锁检测
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;

-- 长时间运行的查询
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
ORDER BY duration DESC;
```

**Python监控脚本：**

```python
# db_monitor.py
import asyncio
import asyncpg
from datetime import datetime, timedelta

class DatabaseMonitor:
    def __init__(self, database_url: str):
        self.database_url = database_url
    
    async def check_long_running_queries(self, threshold_minutes=5):
        """检查长时间运行的查询"""
        conn = await asyncpg.connect(self.database_url)
        
        query = """
        SELECT 
            pid,
            now() - query_start AS duration,
            query,
            state
        FROM pg_stat_activity
        WHERE (now() - query_start) > interval '%s minutes'
        AND state != 'idle'
        ORDER BY duration DESC;
        """ % threshold_minutes
        
        try:
            rows = await conn.fetch(query)
            if rows:
                print(f"发现 {len(rows)} 个长时间运行的查询:")
                for row in rows:
                    print(f"PID: {row['pid']}, 持续时间: {row['duration']}, 查询: {row['query'][:100]}...")
            return rows
        finally:
            await conn.close()
    
    async def check_deadlocks(self):
        """检查死锁"""
        conn = await asyncpg.connect(self.database_url)
        
        query = """
        SELECT 
            blocked_locks.pid AS blocked_pid,
            blocking_locks.pid AS blocking_pid,
            blocked_activity.query AS blocked_query,
            blocking_activity.query AS blocking_query
        FROM pg_catalog.pg_locks blocked_locks
        JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
        JOIN pg_catalog.pg_locks blocking_locks ON (
            blocking_locks.locktype = blocked_locks.locktype
            AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
            AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
            AND blocking_locks.pid != blocked_locks.pid
        )
        JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
        WHERE NOT blocked_locks.granted;
        """
        
        try:
            rows = await conn.fetch(query)
            if rows:
                print(f"发现 {len(rows)} 个死锁:")
                for row in rows:
                    print(f"被阻塞PID: {row['blocked_pid']}, 阻塞PID: {row['blocking_pid']}")
            return rows
        finally:
            await conn.close()
```

## 内存和资源问题

### 内存泄漏检测

```python
# memory_leak_detector.py
import gc
import sys
import tracemalloc
from collections import defaultdict
from typing import Dict, List

class MemoryLeakDetector:
    def __init__(self):
        self.snapshots = []
        self.object_counts = defaultdict(int)
        tracemalloc.start()
    
    def take_snapshot(self, label: str = ""):
        """拍摄内存快照"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
        
        # 统计对象数量
        for obj_type in gc.get_objects():
            type_name = type(obj_type).__name__
            self.object_counts[type_name] += 1
    
    def compare_snapshots(self, start_index: int = 0, end_index: int = -1):
        """比较内存快照"""
        if len(self.snapshots) < 2:
            print("需要至少两个快照进行比较")
            return
        
        start_label, start_snapshot = self.snapshots[start_index]
        end_label, end_snapshot = self.snapshots[end_index]
        
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        print(f"内存变化对比 ({start_label} -> {end_label}):")
        for stat in top_stats[:10]:
            print(stat)
    
    def find_memory_leaks(self) -> List[str]:
        """查找可能的内存泄漏"""
        leaks = []
        
        # 检查引用计数异常高的对象
        for obj in gc.get_objects():
            ref_count = sys.getrefcount(obj)
            if ref_count > 100:  # 阈值可调整
                leaks.append(f"高引用计数对象: {type(obj).__name__} ({ref_count} 引用)")
        
        # 检查循环引用
        gc.collect()
        if gc.garbage:
            leaks.append(f"发现 {len(gc.garbage)} 个无法回收的对象")
        
        return leaks
    
    def get_largest_objects(self, limit: int = 10) -> List[str]:
        """获取占用内存最大的对象"""
        if not self.snapshots:
            return []
        
        _, latest_snapshot = self.snapshots[-1]
        top_stats = latest_snapshot.statistics('lineno')
        
        return [str(stat) for stat in top_stats[:limit]]

# 使用示例
detector = MemoryLeakDetector()

# 在关键点拍摄快照
detector.take_snapshot("开始")
# ... 运行代码 ...
detector.take_snapshot("处理后")

# 分析结果
detector.compare_snapshots()
leaks = detector.find_memory_leaks()
if leaks:
    print("可能的内存泄漏:")
    for leak in leaks:
        print(f"  - {leak}")
```

## 日志分析

### 日志分析工具

```python
# log_analyzer.py
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class LogAnalyzer:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.error_patterns = {
            'timeout': r'timeout|timed out',
            'connection': r'connection.*error|failed to connect',
            'memory': r'out of memory|memory error',
            'api_error': r'api.*error|rate limit|authentication',
            'import_error': r'import.*error|module.*not found'
        }
    
    def analyze_errors(self, hours: int = 24) -> Dict[str, int]:
        """分析错误日志"""
        error_counts = defaultdict(int)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 尝试解析JSON格式日志
                    log_entry = json.loads(line)
                    log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                    
                    if log_time < cutoff_time:
                        continue
                    
                    level = log_entry.get('level', '').lower()
                    message = log_entry.get('message', '')
                    
                    if level in ['error', 'critical']:
                        # 匹配错误模式
                        for error_type, pattern in self.error_patterns.items():
                            if re.search(pattern, message, re.IGNORECASE):
                                error_counts[error_type] += 1
                                break
                        else:
                            error_counts['other'] += 1
                
                except (json.JSONDecodeError, ValueError):
                    # 处理非JSON格式日志
                    if 'ERROR' in line or 'CRITICAL' in line:
                        for error_type, pattern in self.error_patterns.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                error_counts[error_type] += 1
                                break
        
        return dict(error_counts)
    
    def find_performance_issues(self) -> List[str]:
        """查找性能问题"""
        issues = []
        slow_operations = []
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 查找执行时间过长的操作
                duration_match = re.search(r'执行时间[：:]\s*(\d+\.?\d*)秒', line)
                if duration_match:
                    duration = float(duration_match.group(1))
                    if duration > 10:  # 超过10秒
                        slow_operations.append((duration, line.strip()))
        
        if slow_operations:
            slow_operations.sort(reverse=True)
            issues.append(f"发现 {len(slow_operations)} 个慢操作")
            for duration, log_line in slow_operations[:5]:
                issues.append(f"  - {duration:.2f}秒: {log_line[:100]}...")
        
        return issues
    
    def get_error_trends(self, days: int = 7) -> Dict[str, List[int]]:
        """获取错误趋势"""
        trends = defaultdict(lambda: [0] * days)
        
        for day in range(days):
            start_time = datetime.now() - timedelta(days=day+1)
            end_time = datetime.now() - timedelta(days=day)
            
            day_errors = self._count_errors_in_timerange(start_time, end_time)
            
            for error_type, count in day_errors.items():
                trends[error_type][days-1-day] = count
        
        return dict(trends)
    
    def _count_errors_in_timerange(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """统计时间范围内的错误"""
        error_counts = defaultdict(int)
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                    
                    if start_time <= log_time < end_time:
                        level = log_entry.get('level', '').lower()
                        if level in ['error', 'critical']:
                            message = log_entry.get('message', '')
                            for error_type, pattern in self.error_patterns.items():
                                if re.search(pattern, message, re.IGNORECASE):
                                    error_counts[error_type] += 1
                                    break
                
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return dict(error_counts)

# 使用示例
analyzer = LogAnalyzer('logs/agent_workflow.log')

# 分析最近24小时的错误
errors = analyzer.analyze_errors(24)
print("错误统计:", errors)

# 查找性能问题
performance_issues = analyzer.find_performance_issues()
if performance_issues:
    print("性能问题:")
    for issue in performance_issues:
        print(f"  {issue}")

# 获取错误趋势
trends = analyzer.get_error_trends(7)
print("错误趋势:", trends)
```

## 调试工具

### 交互式调试器

```python
# interactive_debugger.py
import asyncio
import sys
import traceback
from typing import Any, Dict

class InteractiveDebugger:
    def __init__(self):
        self.context = {}
        self.history = []
    
    async def debug_session(self, initial_context: Dict[str, Any] = None):
        """启动交互式调试会话"""
        if initial_context:
            self.context.update(initial_context)
        
        print("🐛 交互式调试器启动")
        print("输入 'help' 查看可用命令")
        print("输入 'exit' 退出调试器")
        
        while True:
            try:
                command = input("debug> ").strip()
                
                if command == 'exit':
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'context':
                    self._show_context()
                elif command == 'history':
                    self._show_history()
                elif command.startswith('set '):
                    self._handle_set_command(command)
                elif command.startswith('get '):
                    self._handle_get_command(command)
                elif command.startswith('call '):
                    await self._handle_call_command(command)
                else:
                    # 执行Python代码
                    await self._execute_code(command)
                
                self.history.append(command)
            
            except KeyboardInterrupt:
                print("\n使用 'exit' 退出调试器")
            except Exception as e:
                print(f"错误: {e}")
                traceback.print_exc()
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
  help          - 显示此帮助信息
  context       - 显示当前上下文变量
  history       - 显示命令历史
  set <var> <value> - 设置变量
  get <var>     - 获取变量值
  call <func>   - 调用函数
  exit          - 退出调试器
  
也可以直接输入Python代码执行
        """
        print(help_text)
    
    def _show_context(self):
        """显示上下文"""
        print("当前上下文变量:")
        for key, value in self.context.items():
            print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}")
    
    def _show_history(self):
        """显示历史命令"""
        print("命令历史:")
        for i, cmd in enumerate(self.history[-10:], 1):
            print(f"  {i}: {cmd}")
    
    def _handle_set_command(self, command: str):
        """处理set命令"""
        parts = command.split(' ', 2)
        if len(parts) >= 3:
            var_name = parts[1]
            var_value = ' '.join(parts[2:])
            try:
                # 尝试eval
                self.context[var_name] = eval(var_value, self.context)
                print(f"设置 {var_name} = {self.context[var_name]}")
            except:
                # 作为字符串处理
                self.context[var_name] = var_value
                print(f"设置 {var_name} = '{var_value}'")
        else:
            print("用法: set <变量名> <值>")
    
    def _handle_get_command(self, command: str):
        """处理get命令"""
        parts = command.split(' ', 1)
        if len(parts) >= 2:
            var_name = parts[1]
            if var_name in self.context:
                print(f"{var_name} = {repr(self.context[var_name])}")
            else:
                print(f"变量 '{var_name}' 不存在")
        else:
            print("用法: get <变量名>")
    
    async def _handle_call_command(self, command: str):
        """处理call命令"""
        parts = command.split(' ', 1)
        if len(parts) >= 2:
            func_call = parts[1]
            try:
                result = eval(func_call, self.context)
                if asyncio.iscoroutine(result):
                    result = await result
                print(f"结果: {repr(result)}")
                self.context['_last_result'] = result
            except Exception as e:
                print(f"调用失败: {e}")
        else:
            print("用法: call <函数调用>")
    
    async def _execute_code(self, code: str):
        """执行Python代码"""
        try:
            # 尝试作为表达式执行
            result = eval(code, self.context)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                print(repr(result))
                self.context['_'] = result
        except SyntaxError:
            # 作为语句执行
            try:
                exec(code, self.context)
            except Exception as e:
                print(f"执行错误: {e}")
        except Exception as e:
            print(f"执行错误: {e}")

# 使用示例
async def debug_agent_issue():
    """调试代理问题"""
    from src.core.config.config_manager import ConfigManager
    from src.agents.factory.agent_factory import AgentFactory
    
    # 准备调试上下文
    context = {
        'config_manager': ConfigManager(),
        'agent_factory': AgentFactory(),
        'asyncio': asyncio
    }
    
    debugger = InteractiveDebugger()
    await debugger.debug_session(context)

if __name__ == "__main__":
    asyncio.run(debug_agent_issue())
```

## 常见错误代码

### 错误代码对照表

| 错误代码 | 描述 | 可能原因 | 解决方案 |
|---------|------|----------|----------|
| AGW-001 | 配置文件加载失败 | 文件不存在或格式错误 | 检查配置文件路径和格式 |
| AGW-002 | API密钥无效 | 密钥错误或过期 | 更新API密钥 |
| AGW-003 | 模块导入失败 | Python路径配置问题 | 检查PYTHONPATH设置 |
| AGW-004 | 数据库连接失败 | 连接字符串错误或服务不可用 | 检查数据库配置和服务状态 |
| AGW-005 | 任务执行超时 | 任务复杂度过高或资源不足 | 增加超时时间或优化任务 |
| AGW-006 | 内存不足 | 内存泄漏或数据量过大 | 检查内存使用和优化代码 |
| AGW-007 | 网络连接错误 | 网络问题或代理配置 | 检查网络连接和代理设置 |
| AGW-008 | 权限不足 | 文件或目录权限问题 | 检查文件权限设置 |
| AGW-009 | 依赖版本冲突 | 包版本不兼容 | 更新或降级相关包 |
| AGW-010 | 异步操作错误 | 事件循环问题 | 检查异步代码实现 |

### 自动错误诊断

```python
# auto_diagnosis.py
import sys
import traceback
from typing import Dict, List, Tuple

class AutoDiagnostic:
    def __init__(self):
        self.error_solutions = {
            'ModuleNotFoundError': self._diagnose_import_error,
            'FileNotFoundError': self._diagnose_file_error,
            'ConnectionError': self._diagnose_connection_error,
            'TimeoutError': self._diagnose_timeout_error,
            'MemoryError': self._diagnose_memory_error,
            'PermissionError': self._diagnose_permission_error,
        }
    
    def diagnose_exception(self, exc: Exception) -> List[str]:
        """诊断异常并提供解决建议"""
        exc_type = type(exc).__name__
        suggestions = []
        
        # 通用建议
        suggestions.append(f"错误类型: {exc_type}")
        suggestions.append(f"错误信息: {str(exc)}")
        
        # 特定诊断
        if exc_type in self.error_solutions:
            specific_suggestions = self.error_solutions[exc_type](exc)
            suggestions.extend(specific_suggestions)
        
        # 堆栈跟踪分析
        stack_suggestions = self._analyze_stack_trace(exc)
        suggestions.extend(stack_suggestions)
        
        return suggestions
    
    def _diagnose_import_error(self, exc: ModuleNotFoundError) -> List[str]:
        """诊断导入错误"""
        module_name = str(exc).split("'")[1] if "'" in str(exc) else "unknown"
        
        suggestions = [
            "导入错误诊断:",
            f"  1. 检查模块 '{module_name}' 是否已安装: pip list | grep {module_name}",
            f"  2. 安装缺失模块: pip install {module_name}",
            "  3. 检查PYTHONPATH设置: echo $PYTHONPATH",
            "  4. 检查虚拟环境是否激活",
            "  5. 检查项目结构和__init__.py文件"
        ]
        
        if 'src.' in module_name:
            suggestions.extend([
                "  6. 项目模块导入问题:",
                "     - 确保src目录在Python路径中",
                "     - 运行: export PYTHONPATH=$PYTHONPATH:$(pwd)/src",
                "     - 或安装为可编辑包: pip install -e ."
            ])
        
        return suggestions
    
    def _diagnose_file_error(self, exc: FileNotFoundError) -> List[str]:
        """诊断文件错误"""
        filename = exc.filename or "unknown"
        
        return [
            "文件错误诊断:",
            f"  1. 检查文件是否存在: ls -la {filename}",
            f"  2. 检查文件路径是否正确",
            f"  3. 检查当前工作目录: pwd",
            f"  4. 检查文件权限: ls -la $(dirname {filename})",
            f"  5. 如果是配置文件，检查是否需要创建默认配置"
        ]
    
    def _diagnose_connection_error(self, exc: Exception) -> List[str]:
        """诊断连接错误"""
        return [
            "连接错误诊断:",
            "  1. 检查网络连接: ping google.com",
            "  2. 检查防火墙设置",
            "  3. 检查代理配置: echo $HTTP_PROXY",
            "  4. 检查API端点是否可访问",
            "  5. 检查API密钥是否有效",
            "  6. 检查服务是否正在运行"
        ]
    
    def _diagnose_timeout_error(self, exc: Exception) -> List[str]:
        """诊断超时错误"""
        return [
            "超时错误诊断:",
            "  1. 增加超时时间设置",
            "  2. 检查网络延迟: ping target_host",
            "  3. 检查系统负载: top",
            "  4. 优化代码性能",
            "  5. 考虑使用异步处理",
            "  6. 检查是否有死锁"
        ]
    
    def _diagnose_memory_error(self, exc: MemoryError) -> List[str]:
        """诊断内存错误"""
        return [
            "内存错误诊断:",
            "  1. 检查系统内存使用: free -h",
            "  2. 检查进程内存使用: ps aux | grep python",
            "  3. 查找内存泄漏",
            "  4. 优化数据结构使用",
            "  5. 使用生成器代替列表",
            "  6. 及时释放大对象",
            "  7. 考虑增加系统内存"
        ]
    
    def _diagnose_permission_error(self, exc: PermissionError) -> List[str]:
        """诊断权限错误"""
        return [
            "权限错误诊断:",
            "  1. 检查文件权限: ls -la target_file",
            "  2. 检查目录权限: ls -ld target_directory",
            "  3. 检查用户权限: whoami",
            "  4. 修改文件权限: chmod 644 target_file",
            "  5. 修改目录权限: chmod 755 target_directory",
            "  6. 检查SELinux设置 (Linux)"
        ]
    
    def _analyze_stack_trace(self, exc: Exception) -> List[str]:
        """分析堆栈跟踪"""
        suggestions = []
        tb = traceback.extract_tb(exc.__traceback__)
        
        if tb:
            last_frame = tb[-1]
            suggestions.extend([
                "堆栈跟踪分析:",
                f"  错误发生在: {last_frame.filename}:{last_frame.lineno}",
                f"  函数: {last_frame.name}",
                f"  代码: {last_frame.line}"
            ])
            
            # 检查是否是项目代码
            if 'src/' in last_frame.filename or 'AgentWorker/' in last_frame.filename:
                suggestions.append("  这是项目代码中的错误，建议检查相关逻辑")
            else:
                suggestions.append("  这是第三方库中的错误，可能是使用方式不当")
        
        return suggestions

# 全局异常处理器
def setup_global_exception_handler():
    """设置全局异常处理器"""
    diagnostic = AutoDiagnostic()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print("\n" + "="*50)
        print("🚨 发生未处理的异常")
        print("="*50)
        
        # 显示原始错误
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        print("\n" + "-"*50)
        print("🔧 自动诊断建议")
        print("-"*50)
        
        # 显示诊断建议
        suggestions = diagnostic.diagnose_exception(exc_value)
        for suggestion in suggestions:
            print(suggestion)
        
        print("\n" + "="*50)
    
    sys.excepthook = handle_exception

# 使用示例
if __name__ == "__main__":
    setup_global_exception_handler()
    
    # 测试各种错误
    try:
        import non_existent_module
    except Exception as e:
        diagnostic = AutoDiagnostic()
        suggestions = diagnostic.diagnose_exception(e)
        for suggestion in suggestions:
            print(suggestion)
```

---

通过使用本故障排除指南，您可以快速诊断和解决AI Agents Workflow框架中遇到的各种问题。建议将常用的诊断脚本保存为项目工具，以便在遇到问题时快速使用。