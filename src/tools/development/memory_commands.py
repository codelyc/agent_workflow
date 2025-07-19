# -*- coding: utf-8 -*-
"""
Tools for interacting with the agent's memory.
"""

import json
from ...core.lib.memory import get_memory_manager

def save_memory(key: str, value: str, category: str = "general") -> str:
    """Implementation of the save memory tool."""
    try:
        memory_manager = get_memory_manager()
        memory_manager.set_task_memory(key, category, {"value": value, "timestamp": "now"})
        
        return json.dumps({
            "success": True,
            "key": key,
            "category": category,
            "message": "Memory saved successfully."
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def retrieve_memory(key: str, category: str = "general") -> str:
    """Implementation of the retrieve memory tool."""
    try:
        memory_manager = get_memory_manager()
        data = memory_manager.get_task_memory(key, category)
        
        if data:
            return json.dumps({
                "success": True,
                "key": key,
                "category": category,
                "data": data
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "message": "Memory not found"
            })
            
    except Exception as e:
        return json.dumps({"error": str(e)}) 