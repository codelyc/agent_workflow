"""
分层记忆系统

提供项目级记忆、任务级记忆的分层存储和上下文管理
支持持久化存储和智能检索
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """记忆条目"""
    key: str
    value: Any
    memory_type: str  # project, task, context, temporary
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def update_value(self, new_value: Any, **metadata):
        """更新值"""
        self.value = new_value
        self.updated_at = datetime.now()
        if metadata:
            self.metadata.update(metadata)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换日期时间为字符串
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """从字典创建"""
        # 转换日期时间字符串为datetime对象
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class MemoryStore:
    """记忆存储"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.memories: Dict[str, MemoryEntry] = {}
        self._load_from_storage()
    
    def store(self, key: str, value: Any, memory_type: str = "temporary", 
              expires_at: Optional[datetime] = None, **metadata) -> MemoryEntry:
        """存储记忆"""
        if key in self.memories:
            # 更新现有记忆
            self.memories[key].update_value(value, **metadata)
            entry = self.memories[key]
        else:
            # 创建新记忆
            entry = MemoryEntry(
                key=key,
                value=value,
                memory_type=memory_type,
                expires_at=expires_at,
                metadata=metadata
            )
            self.memories[key] = entry
        
        # 保存到存储
        self._save_to_storage()
        
        logger.debug(f"存储记忆: {key} ({memory_type})")
        return entry
    
    def retrieve(self, key: str) -> Optional[Any]:
        """检索记忆"""
        entry = self.memories.get(key)
        if entry is None:
            return None
        
        # 检查是否过期
        if entry.is_expired():
            self.remove(key)
            return None
        
        logger.debug(f"检索记忆: {key}")
        return entry.value
    
    def get_entry(self, key: str) -> Optional[MemoryEntry]:
        """获取记忆条目"""
        entry = self.memories.get(key)
        if entry and entry.is_expired():
            self.remove(key)
            return None
        return entry
    
    def remove(self, key: str) -> bool:
        """删除记忆"""
        if key in self.memories:
            del self.memories[key]
            self._save_to_storage()
            logger.debug(f"删除记忆: {key}")
            return True
        return False
    
    def list_keys(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[str]:
        """列出记忆键"""
        keys = []
        for key, entry in self.memories.items():
            if entry.is_expired():
                continue
            
            # 按类型过滤
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # 按标签过滤
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            keys.append(key)
        
        return keys
    
    def cleanup_expired(self) -> int:
        """清理过期记忆"""
        expired_keys = [key for key, entry in self.memories.items() if entry.is_expired()]
        for key in expired_keys:
            del self.memories[key]
        
        if expired_keys:
            self._save_to_storage()
            logger.info(f"清理过期记忆: {len(expired_keys)} 条")
        
        return len(expired_keys)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计"""
        total = len(self.memories)
        by_type = {}
        expired = 0
        
        for entry in self.memories.values():
            if entry.is_expired():
                expired += 1
            else:
                by_type[entry.memory_type] = by_type.get(entry.memory_type, 0) + 1
        
        return {
            "total": total,
            "active": total - expired,
            "expired": expired,
            "by_type": by_type
        }
    
    def _load_from_storage(self):
        """从存储加载"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, entry_data in data.items():
                try:
                    self.memories[key] = MemoryEntry.from_dict(entry_data)
                except Exception as e:
                    logger.warning(f"加载记忆条目失败 {key}: {e}")
            
            logger.info(f"从存储加载记忆: {len(self.memories)} 条")
            
        except Exception as e:
            logger.warning(f"加载记忆存储失败: {e}")
    
    def _save_to_storage(self):
        """保存到存储"""
        if not self.storage_path:
            return
        
        try:
            # 确保存储目录存在
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            data = {key: entry.to_dict() for key, entry in self.memories.items()}
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存记忆到存储: {len(self.memories)} 条")
            
        except Exception as e:
            logger.error(f"保存记忆存储失败: {e}")


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        
        # 创建不同层次的记忆存储
        self.project_store = MemoryStore(str(self.workspace_path / ".ai_agents" / "project_memory.json"))
        self.task_store = MemoryStore(str(self.workspace_path / ".ai_agents" / "task_memory.json"))
        self.context_store = MemoryStore()  # 临时存储，不持久化
        
        # 预定义的记忆键
        self.PROJECT_KEYS = {
            "summary": "project.summary",
            "structure": "project.structure", 
            "technology_stack": "project.technology_stack",
            "architecture": "project.architecture",
            "dependencies": "project.dependencies",
            "configuration": "project.configuration"
        }
        
        self.TASK_KEYS = {
            "progress": "task.progress",
            "results": "task.results",
            "context": "task.context",
            "metadata": "task.metadata"
        }
    
    # 项目级记忆管理
    def set_project_memory(self, key: str, value: Any, **metadata) -> MemoryEntry:
        """设置项目记忆"""
        full_key = f"project.{key}" if not key.startswith("project.") else key
        return self.project_store.store(full_key, value, "project", **metadata)
    
    def get_project_memory(self, key: str) -> Optional[Any]:
        """获取项目记忆"""
        full_key = f"project.{key}" if not key.startswith("project.") else key
        return self.project_store.retrieve(full_key)
    
    def set_project_summary(self, summary: str) -> MemoryEntry:
        """设置项目概述"""
        return self.set_project_memory("summary", summary)
    
    def get_project_summary(self) -> Optional[str]:
        """获取项目概述"""
        return self.get_project_memory("summary")
    
    def set_project_structure(self, structure: Dict[str, Any]) -> MemoryEntry:
        """设置项目结构"""
        return self.set_project_memory("structure", structure)
    
    def get_project_structure(self) -> Optional[Dict[str, Any]]:
        """获取项目结构"""
        return self.get_project_memory("structure")
    
    def set_technology_stack(self, tech_stack: List[str]) -> MemoryEntry:
        """设置技术栈"""
        return self.set_project_memory("technology_stack", tech_stack)
    
    def get_technology_stack(self) -> Optional[List[str]]:
        """获取技术栈"""
        return self.get_project_memory("technology_stack")
    
    # 任务级记忆管理
    def set_task_memory(self, task_id: str, key: str, value: Any, **metadata) -> MemoryEntry:
        """设置任务记忆"""
        full_key = f"task.{task_id}.{key}"
        return self.task_store.store(full_key, value, "task", task_id=task_id, **metadata)
    
    def get_task_memory(self, task_id: str, key: str) -> Optional[Any]:
        """获取任务记忆"""
        full_key = f"task.{task_id}.{key}"
        return self.task_store.retrieve(full_key)
    
    def set_task_progress(self, task_id: str, progress: Dict[str, Any]) -> MemoryEntry:
        """设置任务进度"""
        return self.set_task_memory(task_id, "progress", progress)
    
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务进度"""
        return self.get_task_memory(task_id, "progress")
    
    def set_task_results(self, task_id: str, results: Any) -> MemoryEntry:
        """设置任务结果"""
        return self.set_task_memory(task_id, "results", results)
    
    def get_task_results(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self.get_task_memory(task_id, "results")
    
    # 上下文记忆管理
    def set_context_memory(self, key: str, value: Any, expires_at: Optional[datetime] = None) -> MemoryEntry:
        """设置上下文记忆"""
        return self.context_store.store(key, value, "context", expires_at=expires_at)
    
    def get_context_memory(self, key: str) -> Optional[Any]:
        """获取上下文记忆"""
        return self.context_store.retrieve(key)
    
    # 通用操作
    def search_memories(self, pattern: str, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索记忆"""
        results = []
        
        stores = []
        if memory_type == "project":
            stores = [self.project_store]
        elif memory_type == "task":
            stores = [self.task_store]
        elif memory_type == "context":
            stores = [self.context_store]
        else:
            stores = [self.project_store, self.task_store, self.context_store]
        
        for store in stores:
            for key in store.list_keys():
                if pattern.lower() in key.lower():
                    entry = store.get_entry(key)
                    if entry:
                        results.append({
                            "key": key,
                            "value": entry.value,
                            "type": entry.memory_type,
                            "created_at": entry.created_at.isoformat(),
                            "updated_at": entry.updated_at.isoformat()
                        })
        
        return results
    
    def cleanup_expired(self) -> Dict[str, int]:
        """清理过期记忆"""
        return {
            "project": self.project_store.cleanup_expired(),
            "task": self.task_store.cleanup_expired(), 
            "context": self.context_store.cleanup_expired()
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "project": self.project_store.get_statistics(),
            "task": self.task_store.get_statistics(),
            "context": self.context_store.get_statistics()
        }
    
    def clear_task_memories(self, task_id: str) -> int:
        """清理特定任务的记忆"""
        prefix = f"task.{task_id}."
        keys_to_remove = [key for key in self.task_store.memories.keys() if key.startswith(prefix)]
        
        for key in keys_to_remove:
            self.task_store.remove(key)
        
        logger.info(f"清理任务记忆: {task_id}, {len(keys_to_remove)} 条")
        return len(keys_to_remove)


# 全局记忆管理器实例
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(workspace_path: Optional[str] = None) -> MemoryManager:
    """获取全局记忆管理器"""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(workspace_path)
    
    return _global_memory_manager


def reset_memory_manager():
    """重置全局记忆管理器"""
    global _global_memory_manager
    _global_memory_manager = None 