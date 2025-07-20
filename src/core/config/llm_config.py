"""
LLM追踪配置管理

管理Langfuse等监控平台的配置和集成
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LangfuseConfig:
    """Langfuse配置"""
    public_key: str
    private_key: str
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True


@dataclass
class TracingConfig:
    """追踪配置"""
    provider: str = "internal"  # internal, langfuse
    enabled: bool = True
    langfuse: Optional[LangfuseConfig] = None
    
    # 内部追踪配置
    auto_trace_llm: bool = True
    trace_input: bool = True
    trace_output: bool = True
    trace_metadata: bool = True


def load_tracing_config() -> TracingConfig:
    """加载追踪配置"""
    config = TracingConfig()
    
    # 从环境变量加载
    config.enabled = os.getenv("TRACING_ENABLED", "true").lower() == "true"
    config.provider = os.getenv("TRACING_PROVIDER", "internal")
    config.auto_trace_llm = os.getenv("AUTO_TRACE_LLM", "true").lower() == "true"
    
    # Langfuse配置
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_private_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if langfuse_public_key and langfuse_private_key:
        config.langfuse = LangfuseConfig(
            public_key=langfuse_public_key,
            private_key=langfuse_private_key,
            host=langfuse_host,
            enabled=True
        )
    
    return config


def setup_tracing_from_config(config: Optional[TracingConfig] = None):
    """根据配置设置追踪"""
    if config is None:
        config = load_tracing_config()
    
    if not config.enabled:
        logger.info("追踪功能已禁用")
        return
    
    from ..tracing import enable_llm_tracing
    
    if config.provider == "langfuse" and config.langfuse:
        # 启用Langfuse追踪
        enable_llm_tracing(
            provider="langfuse",
            public_key=config.langfuse.public_key,
            private_key=config.langfuse.private_key,
            host=config.langfuse.host
        )
        logger.info("已启用Langfuse追踪")
    else:
        # 启用内部追踪
        enable_llm_tracing(provider="internal")
        logger.info("已启用内部追踪")


def get_langfuse_config() -> Optional[LangfuseConfig]:
    """获取Langfuse配置"""
    config = load_tracing_config()
    return config.langfuse


def setup_langfuse_env(public_key: str, private_key: str, host: Optional[str] = None):
    """设置Langfuse环境变量"""
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = private_key
    if host:
        os.environ["LANGFUSE_HOST"] = host
    
    logger.info("Langfuse环境变量已设置")


def clear_langfuse_env():
    """清除Langfuse环境变量"""
    env_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    logger.info("Langfuse环境变量已清除")
 