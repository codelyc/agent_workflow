"""
多层追踪系统

提供完整的任务追踪、链路追踪、性能监控等功能
支持主任务ID、子任务ID的分层管理
支持LLM调用自动追踪和监控平台集成
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

# 第三方库
import litellm
from langfuse import Langfuse

# 上下文变量
_current_task_id: ContextVar[Optional[str]] = ContextVar('current_task_id', default=None)
_current_sub_task_id: ContextVar[Optional[str]] = ContextVar('current_sub_task_id', default=None)
_current_agent_id: ContextVar[Optional[str]] = ContextVar('current_agent_id', default=None)

# 追踪回调函数列表
_trace_callbacks: List[Callable[[Dict], None]] = []

# LLM追踪配置
_llm_tracing_enabled: bool = False
_original_completion: Optional[Callable] = None

# 日志配置
logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """追踪事件"""
    event_type: str  # task_start, task_end, tool_call, llm_call, error等
    category: str    # workflow, agent, tool, llm等
    name: str        # 具体的操作名称
    
    # 追踪标识
    task_id: Optional[str] = None
    sub_task_id: Optional[str] = None
    agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None
    
    # 事件数据
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 状态信息
    status: str = "success"  # success, error, pending
    error_message: Optional[str] = None


class TraceManager:
    """追踪管理器"""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.active_spans: Dict[str, TraceEvent] = {}
    
    def add_event(self, event: TraceEvent):
        """添加追踪事件"""
        self.events.append(event)
        
        # 触发回调函数
        for callback in _trace_callbacks:
            try:
                callback(event.__dict__)
            except Exception as e:
                logger.warning(f"追踪回调执行失败: {e}")
    
    def start_span(self, span_id: str, event: TraceEvent):
        """开始一个span"""
        self.active_spans[span_id] = event
        self.add_event(event)
    
    def end_span(self, span_id: str, output_data: Optional[Dict] = None, 
                 status: str = "success", error_message: Optional[str] = None):
        """结束一个span"""
        if span_id in self.active_spans:
            event = self.active_spans[span_id]
            event.duration = (datetime.now() - event.timestamp).total_seconds()
            event.output_data = output_data
            event.status = status
            event.error_message = error_message
            
            # 创建结束事件
            end_event = TraceEvent(
                event_type=f"{event.event_type}_end",
                category=event.category,
                name=event.name,
                task_id=event.task_id,
                sub_task_id=event.sub_task_id,
                agent_id=event.agent_id,
                trace_id=event.trace_id,
                duration=event.duration,
                output_data=output_data,
                metadata=event.metadata,
                status=status,
                error_message=error_message
            )
            
            self.add_event(end_event)
            del self.active_spans[span_id]


# 全局追踪管理器
_trace_manager = TraceManager()


def get_current_task_id() -> Optional[str]:
    """获取当前任务ID"""
    return _current_task_id.get()


def get_current_sub_task_id() -> Optional[str]:
    """获取当前子任务ID"""
    return _current_sub_task_id.get()


def get_current_agent_id() -> Optional[str]:
    """获取当前智能体ID"""
    return _current_agent_id.get()


def get_current_trace_chain() -> str:
    """获取当前完整的追踪链"""
    chain = []
    if task_id := get_current_task_id():
        chain.append(task_id)
    if sub_task_id := get_current_sub_task_id():
        chain.append(sub_task_id)
    if agent_id := get_current_agent_id():
        chain.append(agent_id)
    return ".".join(chain) if chain else ""


def add_trace_callback(callback: Callable[[Dict], None]):
    """添加追踪回调函数"""
    _trace_callbacks.append(callback)


def remove_trace_callback(callback: Callable[[Dict], None]):
    """移除追踪回调函数"""
    if callback in _trace_callbacks:
        _trace_callbacks.remove(callback)


@contextmanager
def task_context(task_id: Optional[str] = None):
    """任务上下文管理器"""
    if task_id is None:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # 创建任务开始事件
    start_event = TraceEvent(
        event_type="task_start",
        category="workflow",
        name="task_execution",
        task_id=task_id,
        trace_id=task_id
    )
    _trace_manager.add_event(start_event)
    
    # 设置上下文变量
    token = _current_task_id.set(task_id)
    
    try:
        yield task_id
    except Exception as e:
        # 记录错误事件
        error_event = TraceEvent(
            event_type="task_error",
            category="workflow",
            name="task_execution",
            task_id=task_id,
            trace_id=task_id,
            status="error",
            error_message=str(e)
        )
        _trace_manager.add_event(error_event)
        raise
    finally:
        # 创建任务结束事件
        end_event = TraceEvent(
            event_type="task_end",
            category="workflow", 
            name="task_execution",
            task_id=task_id,
            trace_id=task_id
        )
        _trace_manager.add_event(end_event)
        
        # 重置上下文变量
        _current_task_id.reset(token)


@contextmanager
def sub_task_context(sub_task_id: Optional[str] = None):
    """子任务上下文管理器"""
    if sub_task_id is None:
        sub_task_id = f"subtask_{uuid.uuid4().hex[:8]}"
    
    task_id = get_current_task_id()
    trace_id = f"{task_id}.{sub_task_id}" if task_id else sub_task_id
    
    # 创建子任务开始事件
    start_event = TraceEvent(
        event_type="subtask_start",
        category="agent",
        name="subtask_execution",
        task_id=task_id,
        sub_task_id=sub_task_id,
        trace_id=trace_id
    )
    _trace_manager.add_event(start_event)
    
    # 设置上下文变量
    token = _current_sub_task_id.set(sub_task_id)
    
    try:
        yield sub_task_id
    except Exception as e:
        # 记录错误事件
        error_event = TraceEvent(
            event_type="subtask_error",
            category="agent",
            name="subtask_execution",
            task_id=task_id,
            sub_task_id=sub_task_id,
            trace_id=trace_id,
            status="error",
            error_message=str(e)
        )
        _trace_manager.add_event(error_event)
        raise
    finally:
        # 创建子任务结束事件
        end_event = TraceEvent(
            event_type="subtask_end",
            category="agent",
            name="subtask_execution",
            task_id=task_id,
            sub_task_id=sub_task_id,
            trace_id=trace_id
        )
        _trace_manager.add_event(end_event)
        
        # 重置上下文变量
        _current_sub_task_id.reset(token)


@contextmanager
def agent_context(agent_id: str):
    """智能体上下文管理器"""
    task_id = get_current_task_id()
    sub_task_id = get_current_sub_task_id()
    
    # 构建完整的追踪ID
    trace_components = [task_id, sub_task_id, agent_id]
    trace_id = ".".join([c for c in trace_components if c])
    
    # 创建智能体开始事件
    start_event = TraceEvent(
        event_type="agent_start",
        category="agent",
        name=agent_id,
        task_id=task_id,
        sub_task_id=sub_task_id,
        agent_id=agent_id,
        trace_id=trace_id
    )
    _trace_manager.add_event(start_event)
    
    # 设置上下文变量
    token = _current_agent_id.set(agent_id)
    
    try:
        yield agent_id
    except Exception as e:
        # 记录错误事件
        error_event = TraceEvent(
            event_type="agent_error",
            category="agent",
            name=agent_id,
            task_id=task_id,
            sub_task_id=sub_task_id,
            agent_id=agent_id,
            trace_id=trace_id,
            status="error",
            error_message=str(e)
        )
        _trace_manager.add_event(error_event)
        raise
    finally:
        # 创建智能体结束事件
        end_event = TraceEvent(
            event_type="agent_end",
            category="agent",
            name=agent_id,
            task_id=task_id,
            sub_task_id=sub_task_id,
            agent_id=agent_id,
            trace_id=trace_id
        )
        _trace_manager.add_event(end_event)
        
        # 重置上下文变量
        _current_agent_id.reset(token)


@contextmanager
def trace_event(event_type: str, category: str, name: str = "", **metadata):
    """通用事件追踪上下文管理器"""
    span_id = f"{event_type}_{uuid.uuid4().hex[:8]}"
    
    # 获取当前追踪信息
    task_id = get_current_task_id()
    sub_task_id = get_current_sub_task_id()
    agent_id = get_current_agent_id()
    trace_id = get_current_trace_chain()
    
    # 创建开始事件
    start_event = TraceEvent(
        event_type=f"{event_type}_start",
        category=category,
        name=name or event_type,
        task_id=task_id,
        sub_task_id=sub_task_id,
        agent_id=agent_id,
        trace_id=trace_id,
        metadata=metadata
    )
    
    _trace_manager.start_span(span_id, start_event)
    
    try:
        yield span_id
    except Exception as e:
        _trace_manager.end_span(span_id, status="error", error_message=str(e))
        raise
    else:
        _trace_manager.end_span(span_id, status="success")


def _setup_llm_tracing():
    """设置LLM调用自动追踪"""
    global _llm_tracing_enabled, _original_completion
    
    if _llm_tracing_enabled:
        return
    
    try:
        # 保存原始函数
        _original_completion = litellm.completion
        
        def completion_with_tracing(*args, **kwargs):
            """带追踪的LLM completion函数"""
            # 获取当前追踪信息
            task_id = get_current_task_id()
            sub_task_id = get_current_sub_task_id()
            agent_id = get_current_agent_id()
            trace_chain = get_current_trace_chain()
            
            if task_id:
                # 注入追踪信息到metadata中
                if 'metadata' not in kwargs:
                    kwargs['metadata'] = {}
                
                # 添加追踪信息
                kwargs['metadata'].update({
                    'task_id': task_id,
                    'session_id': task_id,  # 同时作为session_id
                    'trace_chain': trace_chain
                })
                
                # 添加子任务信息（如果存在）
                if sub_task_id:
                    kwargs['metadata']['sub_task_id'] = sub_task_id
                    kwargs['metadata']['trace_id'] = f"{task_id}.{sub_task_id}"
                
                # 添加智能体信息（如果存在）
                if agent_id:
                    kwargs['metadata']['agent_id'] = agent_id
                    kwargs['metadata']['user_id'] = agent_id  # 也可以作为user_id
                
                logger.debug(f"注入追踪信息到LLM调用: {kwargs['metadata']}")
            
            # 创建LLM调用追踪事件
            with trace_event("llm_call", "llm", name=kwargs.get('model', 'unknown')) as span_id:
                # 记录输入信息
                input_data = {
                    'model': kwargs.get('model'),
                    'messages': kwargs.get('messages', []),
                    'temperature': kwargs.get('temperature'),
                    'max_tokens': kwargs.get('max_tokens')
                }
                
                start_time = time.time()
                
                try:
                    # 调用原始函数
                    result = _original_completion(*args, **kwargs)
                    
                    # 记录输出信息
                    output_data = {
                        'usage': getattr(result, 'usage', None),
                        'model': getattr(result, 'model', None),
                        'response_time': time.time() - start_time
                    }
                    
                    # 更新span信息
                    if span_id in _trace_manager.active_spans:
                        _trace_manager.active_spans[span_id].input_data = input_data
                        _trace_manager.active_spans[span_id].output_data = output_data
                    
                    return result
                    
                except Exception as e:
                    # 记录错误信息
                    if span_id in _trace_manager.active_spans:
                        _trace_manager.active_spans[span_id].input_data = input_data
                        _trace_manager.active_spans[span_id].status = "error"
                        _trace_manager.active_spans[span_id].error_message = str(e)
                    raise
        
        # 替换litellm.completion函数
        litellm.completion = completion_with_tracing
        _llm_tracing_enabled = True
        
        logger.info("LLM调用自动追踪已启用")
        
    except Exception as e:
        logger.error(f"设置LLM调用追踪失败: {e}")


def _setup_litellm_tracing():
    """设置litellm集成追踪"""
    global _llm_tracing_enabled, _original_completion
    
    if _llm_tracing_enabled:
        return
    
    try:
        # 设置litellm回调来自动追踪
        def setup_litellm_callbacks():
            """设置litellm的成功和失败回调"""
            
            # 自定义成功回调
            def success_callback(kwargs, completion_response, start_time, end_time):
                """LLM调用成功回调"""
                try:
                    # 创建成功事件
                    task_id = get_current_task_id()
                    sub_task_id = get_current_sub_task_id()
                    agent_id = get_current_agent_id()
                    trace_id = get_current_trace_chain()
                    
                    success_event = TraceEvent(
                        event_type="llm_success",
                        category="llm",
                        name=kwargs.get('model', 'unknown'),
                        task_id=task_id,
                        sub_task_id=sub_task_id,
                        agent_id=agent_id,
                        trace_id=trace_id,
                        duration=end_time - start_time,
                        input_data={
                            'model': kwargs.get('model'),
                            'messages': kwargs.get('messages', []),
                            'temperature': kwargs.get('temperature'),
                            'max_tokens': kwargs.get('max_tokens')
                        },
                        output_data={
                            'usage': getattr(completion_response, 'usage', None),
                            'model': getattr(completion_response, 'model', None),
                            'response_time': end_time - start_time
                        },
                        metadata=kwargs.get('metadata', {}),
                        status="success"
                    )
                    
                    _trace_manager.add_event(success_event)
                    logger.debug(f"LLM调用成功: {kwargs.get('model')} - {end_time - start_time:.3f}s")
                    
                except Exception as e:
                    logger.warning(f"LLM成功回调执行失败: {e}")
            
            # 自定义失败回调
            def failure_callback(kwargs, completion_response, start_time, end_time):
                """LLM调用失败回调"""
                try:
                    # 创建失败事件
                    task_id = get_current_task_id()
                    sub_task_id = get_current_sub_task_id()
                    agent_id = get_current_agent_id()
                    trace_id = get_current_trace_chain()
                    
                    error_message = str(completion_response) if completion_response else "Unknown error"
                    
                    failure_event = TraceEvent(
                        event_type="llm_failure",
                        category="llm",
                        name=kwargs.get('model', 'unknown'),
                        task_id=task_id,
                        sub_task_id=sub_task_id,
                        agent_id=agent_id,
                        trace_id=trace_id,
                        duration=end_time - start_time,
                        input_data={
                            'model': kwargs.get('model'),
                            'messages': kwargs.get('messages', []),
                            'temperature': kwargs.get('temperature'),
                            'max_tokens': kwargs.get('max_tokens')
                        },
                        metadata=kwargs.get('metadata', {}),
                        status="error",
                        error_message=error_message
                    )
                    
                    _trace_manager.add_event(failure_event)
                    logger.error(f"LLM调用失败: {kwargs.get('model')} - {error_message}")
                    
                except Exception as e:
                    logger.warning(f"LLM失败回调执行失败: {e}")
            
            # 注册回调到litellm
            if not hasattr(litellm, '_success_callbacks'):
                litellm._success_callbacks = []
            if not hasattr(litellm, '_failure_callbacks'):
                litellm._failure_callbacks = []
                
            # 添加我们的回调
            litellm._success_callbacks.append(success_callback)
            litellm._failure_callbacks.append(failure_callback)
            
            return success_callback, failure_callback
        
        # 设置任务ID注入
        def setup_task_id_injection():
            """设置litellm的预处理回调来自动注入任务ID和子任务信息"""
            _original_completion = litellm.completion

            def completion_with_task_id(*args, **kwargs):
                # 获取当前追踪信息
                task_id = get_current_task_id()
                sub_task_id = get_current_sub_task_id()
                agent_id = get_current_agent_id()

                if task_id:
                    # 注入追踪信息到metadata中
                    if 'metadata' not in kwargs:
                        kwargs['metadata'] = {}

                    # 添加主任务ID
                    kwargs['metadata']['task_id'] = task_id
                    kwargs['metadata']['session_id'] = task_id  # 同时作为session_id

                    # 添加子任务信息（如果存在）
                    if sub_task_id:
                        kwargs['metadata']['sub_task_id'] = sub_task_id
                        kwargs['metadata']['trace_id'] = f"{task_id}.{sub_task_id}"  # 层次化trace_id

                    # 添加智能体信息（如果存在）
                    if agent_id:
                        kwargs['metadata']['agent_id'] = agent_id
                        kwargs['metadata']['user_id'] = agent_id  # 也可以作为user_id用于区分

                    # 构建完整的追踪链路标识
                    trace_chain = [task_id]
                    if sub_task_id:
                        trace_chain.append(sub_task_id)
                    if agent_id:
                        trace_chain.append(agent_id)
                    kwargs['metadata']['trace_chain'] = ".".join(trace_chain)

                    logger.debug(f"注入追踪信息到LLM调用: {kwargs['metadata']}")

                return _original_completion(*args, **kwargs)

            # 替换litellm.completion函数
            litellm.completion = completion_with_task_id
        
        # 执行设置
        setup_task_id_injection()
        setup_litellm_callbacks()
        
        _llm_tracing_enabled = True
        logger.info("litellm集成追踪已启用")
        
    except Exception as e:
        logger.error(f"设置litellm集成追踪失败: {e}")


def enable_langfuse_tracing(public_key: str = None, private_key: str = None, host: str = None):
    """启用Langfuse追踪集成"""
    try:
        # 设置环境变量
        if public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        if private_key:
            os.environ["LANGFUSE_SECRET_KEY"] = private_key
        if host:
            os.environ["LANGFUSE_HOST"] = host
        
        # 设置litellm的Langfuse回调
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
        
        # 设置任务ID注入
        _setup_litellm_tracing()
        
        logger.info("Langfuse追踪集成已启用")
        
    except Exception as e:
        logger.error(f"启用Langfuse追踪失败: {e}")


def enable_llm_tracing(provider: str = "internal", **config):
    """启用LLM调用自动追踪
    
    Args:
        provider: 追踪提供商 ("internal", "langfuse")
        **config: 配置参数
            - public_key: Langfuse公钥
            - private_key: Langfuse私钥  
            - host: Langfuse主机地址
    """
    if provider == "langfuse":
        enable_langfuse_tracing(
            public_key=config.get('public_key'),
            private_key=config.get('private_key'),
            host=config.get('host')
        )
    else:
        # 使用内部追踪
        _setup_litellm_tracing()


def disable_llm_tracing():
    """禁用LLM调用自动追踪"""
    global _llm_tracing_enabled, _original_completion
    
    if not _llm_tracing_enabled:
        return
    
    try:
        # 恢复原始函数
        if _original_completion:
            litellm.completion = _original_completion
            _original_completion = None
        
        # 清除回调
        litellm.success_callback = []
        litellm.failure_callback = []
        
        # 清除自定义回调
        if hasattr(litellm, '_success_callbacks'):
            litellm._success_callbacks.clear()
        if hasattr(litellm, '_failure_callbacks'):
            litellm._failure_callbacks.clear()
        
        _llm_tracing_enabled = False
        logger.info("LLM调用自动追踪已禁用")
        
    except Exception as e:
        logger.error(f"禁用LLM调用追踪失败: {e}")


def _restore_llm_tracing():
    """恢复原始的LLM函数（向后兼容）"""
    disable_llm_tracing()


def get_trace_events() -> List[TraceEvent]:
    """获取所有追踪事件"""
    return _trace_manager.events.copy()


def clear_trace_events():
    """清除所有追踪事件"""
    _trace_manager.events.clear()
    _trace_manager.active_spans.clear()


def get_trace_statistics() -> Dict[str, Any]:
    """获取追踪统计信息"""
    events = _trace_manager.events
    
    stats = {
        'total_events': len(events),
        'events_by_type': {},
        'events_by_category': {},
        'average_duration': 0,
        'total_duration': 0,
        'error_count': 0,
        'success_count': 0
    }
    
    total_duration = 0
    duration_count = 0
    
    for event in events:
        # 按类型统计
        event_type = event.event_type
        stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
        
        # 按类别统计
        category = event.category
        stats['events_by_category'][category] = stats['events_by_category'].get(category, 0) + 1
        
        # 状态统计
        if event.status == "error":
            stats['error_count'] += 1
        else:
            stats['success_count'] += 1
        
        # 持续时间统计
        if event.duration is not None:
            total_duration += event.duration
            duration_count += 1
    
    if duration_count > 0:
        stats['average_duration'] = total_duration / duration_count
        stats['total_duration'] = total_duration
    
    return stats 