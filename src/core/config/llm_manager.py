"""
LLM 配置管理器

负责加载和管理各种 LLM 提供商的配置，包括 OpenAI、Anthropic、本地模型等
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM 提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    ZHIPUAI = "zhipuai"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: LLMProvider
    max_tokens: int
    temperature: float
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    description: str = ""
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


@dataclass
class CostTracker:
    """成本追踪器"""
    daily_spending: float = 0.0
    total_spending: float = 0.0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    request_count: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    def reset_daily_if_needed(self):
        """如果是新的一天，重置每日花费"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_spending = 0.0
            self.last_reset_date = today
    
    def add_cost(self, cost: float, input_tokens: int, output_tokens: int):
        """添加成本记录"""
        self.reset_daily_if_needed()
        self.daily_spending += cost
        self.total_spending += cost
        self.request_count += 1
        self.token_usage["input"] = self.token_usage.get("input", 0) + input_tokens
        self.token_usage["output"] = self.token_usage.get("output", 0) + output_tokens


class SimpleLLMClient:
    """简单的LLM客户端实现"""
    
    def __init__(self, manager: 'LLMManager'):
        self.manager = manager
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """生成文本"""
        # 这里应该实现实际的LLM调用逻辑
        # 目前返回一个模拟响应
        model_name = model or "gpt-4o-mini"
        return f"模拟LLM响应 (模型: {model_name})\n\n基于提示: {prompt[:100]}..."
    
    async def chat_completions_create(self, model: str, messages: List[Dict], **kwargs) -> Any:
        """创建聊天完成"""
        # 模拟OpenAI API响应结构
        class MockChoice:
            def __init__(self, content: str):
                self.message = type('Message', (), {'content': content})()
        
        class MockResponse:
            def __init__(self, content: str):
                self.choices = [MockChoice(content)]
        
        prompt = "\n".join([msg.get('content', '') for msg in messages])
        content = f"模拟聊天响应 (模型: {model})\n\n基于消息: {prompt[:100]}..."
        return MockResponse(content)


class LLMManager:
    """LLM 配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("configs/llm_config.yaml")
        self.config: Dict[str, Any] = {}
        self.models: Dict[str, ModelConfig] = {}
        self.cost_tracker = CostTracker()
        self.response_cache: Dict[str, Any] = {}
        self._client = None
        self.load_config()
    
    def load_config(self):
        """加载 LLM 配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self._parse_models()
                logger.info(f"LLM 配置加载成功: {len(self.models)} 个模型")
            else:
                logger.warning(f"LLM 配置文件不存在: {self.config_path}")
                self._create_default_config()
        except Exception as e:
            logger.error(f"加载 LLM 配置失败: {e}")
            self._create_default_config()
    
    def _parse_models(self):
        """解析模型配置"""
        self.models.clear()
        
        # 元数据字段，跳过处理
        metadata_keys = [
            "config_type", "version", "created_at", "updated_at", 
            "default_provider", "model_aliases", "task_model_mapping", 
            "cost_control", "load_balancing", "caching", "resilience", 
            "monitoring", "development"
        ]
        
        # 解析各个提供商的模型
        for provider_name, provider_config in self.config.items():
            if provider_name in metadata_keys:
                continue
                
            # 确保provider_config是字典类型
            if not isinstance(provider_config, dict):
                logger.warning(f"提供商配置不是字典类型: {provider_name}")
                continue
                
            try:
                provider = LLMProvider(provider_name)
                models = provider_config.get("models", [])
                
                for model_info in models:
                    if not isinstance(model_info, dict):
                        continue
                        
                    model_config = ModelConfig(
                        name=model_info["name"],
                        provider=provider,
                        max_tokens=model_info.get("max_tokens", 4096),
                        temperature=model_info.get("temperature", 0.7),
                        cost_per_1k_input=model_info.get("cost_per_1k_input", 0.0),
                        cost_per_1k_output=model_info.get("cost_per_1k_output", 0.0),
                        description=model_info.get("description", ""),
                        deployment_name=model_info.get("deployment_name"),
                        base_url=provider_config.get("base_url"),
                        api_key=self._resolve_env_var(provider_config.get("api_key", "")),
                        timeout=provider_config.get("timeout", 60),
                        max_retries=provider_config.get("max_retries", 3)
                    )
                    
                    self.models[model_info["name"]] = model_config
                    
                    # 处理别名
                    for alias, model_name in self.config.get("model_aliases", {}).items():
                        if model_name == model_info["name"]:
                            self.models[alias] = model_config
                            
            except ValueError:
                logger.warning(f"未知的 LLM 提供商: {provider_name}")
            except Exception as e:
                logger.error(f"解析 {provider_name} 模型配置失败: {e}")
    
    def _resolve_env_var(self, value: str) -> str:
        """解析环境变量"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_name = value[2:-1]
            return os.getenv(env_name) or ""
        return value or ""
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config = {
            "default_provider": "openai",
            "model_aliases": {
                "fast": "gpt-4o-mini",
                "balanced": "gpt-4o",
                "powerful": "gpt-4o"
            }
        }
        
        # 创建默认模型
        default_model = ModelConfig(
            name="gpt-4o-mini",
            provider=LLMProvider.OPENAI,
            max_tokens=128000,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY") or ""
        )
        self.models["gpt-4o-mini"] = default_model
        self.models["fast"] = default_model
    
    def get_client(self) -> SimpleLLMClient:
        """获取LLM客户端"""
        if self._client is None:
            self._client = SimpleLLMClient(self)
        return self._client
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models.get(model_name)
    
    def get_default_model(self, task_type: str = None) -> Optional[ModelConfig]:
        """获取默认模型"""
        if task_type and task_type in self.config.get("task_model_mapping", {}):
            model_name = self.config["task_model_mapping"][task_type]["default"]
            model_config = self.get_model_config(model_name)
            if model_config:
                return model_config
        
        # 返回默认模型
        default_provider = self.config.get("default_provider", "openai")
        if default_provider == "openai":
            return self.get_model_config("gpt-4o-mini") or list(self.models.values())[0]
        
        return list(self.models.values())[0] if self.models else None
    
    def get_fallback_models(self, task_type: str) -> List[ModelConfig]:
        """获取后备模型列表"""
        fallback_models = []
        
        if task_type in self.config.get("task_model_mapping", {}):
            fallback_names = self.config["task_model_mapping"][task_type].get("fallback", [])
            for name in fallback_names:
                model_config = self.get_model_config(name)
                if model_config:
                    fallback_models.append(model_config)
        
        return fallback_models
    
    def check_budget(self, estimated_cost: float) -> bool:
        """检查预算限制"""
        self.cost_tracker.reset_daily_if_needed()
        
        cost_config = self.config.get("cost_control", {})
        daily_budget = cost_config.get("daily_budget", float('inf'))
        
        return (self.cost_tracker.daily_spending + estimated_cost) <= daily_budget
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """估算成本"""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
        
        return input_cost + output_cost
    
    def record_usage(self, model_name: str, input_tokens: int, output_tokens: int):
        """记录使用情况"""
        cost = self.estimate_cost(model_name, input_tokens, output_tokens)
        self.cost_tracker.add_cost(cost, input_tokens, output_tokens)
        
        if self.config.get("monitoring", {}).get("log_costs", False):
            logger.info(f"模型使用: {model_name}, 输入: {input_tokens}, 输出: {output_tokens}, 成本: ${cost:.4f}")
    
    def get_cache_key(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {
            "prompt": prompt[:500],  # 只使用前500个字符
            "model": model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """获取缓存的响应"""
        if not self.config.get("caching", {}).get("enabled", False):
            return None
        
        cached_item = self.response_cache.get(cache_key)
        if cached_item:
            # 检查是否过期
            expiry_seconds = self.config.get("caching", {}).get("expiry_seconds", 3600)
            if datetime.now().timestamp() - cached_item["timestamp"] < expiry_seconds:
                return cached_item["response"]
            else:
                # 删除过期缓存
                del self.response_cache[cache_key]
        
        return None
    
    def cache_response(self, cache_key: str, response: Any):
        """缓存响应"""
        if not self.config.get("caching", {}).get("enabled", False):
            return
        
        max_entries = self.config.get("caching", {}).get("max_entries", 1000)
        
        # 如果缓存满了，删除最旧的条目
        if len(self.response_cache) >= max_entries:
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k]["timestamp"])
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now().timestamp()
        }
    
    def list_available_models(self, provider: Optional[LLMProvider] = None) -> List[str]:
        """列出可用模型"""
        if provider:
            return [name for name, config in self.models.items() 
                   if config.provider == provider]
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return {}
        
        return {
            "name": model_config.name,
            "provider": model_config.provider.value,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "cost_per_1k_input": model_config.cost_per_1k_input,
            "cost_per_1k_output": model_config.cost_per_1k_output,
            "description": model_config.description
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计"""
        self.cost_tracker.reset_daily_if_needed()
        
        return {
            "daily_spending": self.cost_tracker.daily_spending,
            "total_spending": self.cost_tracker.total_spending,
            "request_count": self.cost_tracker.request_count,
            "token_usage": self.cost_tracker.token_usage,
            "last_reset_date": self.cost_tracker.last_reset_date,
            "available_models": len(self.models),
            "cache_size": len(self.response_cache)
        }
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            "config": self.config,
            "available_models": self.list_available_models(),
            "usage_statistics": self.get_usage_statistics()
        }
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        issues = []
        
        # 检查必需的环境变量
        for provider_name, provider_config in self.config.items():
            if provider_name in ["default_provider", "model_aliases", "task_model_mapping"]:
                continue
                
            api_key = provider_config.get("api_key", "")
            if isinstance(api_key, str) and api_key.startswith("${"):
                env_name = api_key[2:-1]
                if not os.getenv(env_name):
                    issues.append(f"环境变量 {env_name} 未设置 (用于 {provider_name})")
        
        # 检查模型配置
        if not self.models:
            issues.append("没有配置任何可用模型")
        
        # 检查默认模型
        default_model = self.get_default_model()
        if not default_model:
            issues.append("无法获取默认模型")
        
        return issues


# 全局 LLM 管理器实例
_global_llm_manager: Optional[LLMManager] = None


def get_llm_manager(config_path: Optional[str] = None) -> LLMManager:
    """获取全局 LLM 管理器"""
    global _global_llm_manager
    
    if _global_llm_manager is None:
        _global_llm_manager = LLMManager(config_path)
    
    return _global_llm_manager


def reset_llm_manager():
    """重置全局 LLM 管理器"""
    global _global_llm_manager
    _global_llm_manager = None