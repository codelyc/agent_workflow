"""
智能体工厂

基于YAML配置创建和管理智能体实例
"""

import logging
from typing import Dict, Optional, Union, Any
from pathlib import Path

from ..supervisor.smolagents_supervisor import SmolaAgentsSupervisor
from ..micro.smolagents_micro import SmolaAgentsMicro
from core.config.config_manager import ConfigManager
from core.config.agent_config import AgentConfig, AgentType

logger = logging.getLogger(__name__)


class AgentFactory:
    """智能体工厂类"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self._agent_cache: Dict[str, Union[SmolaAgentsSupervisor, SmolaAgentsMicro]] = {}
        
        # 确保配置已加载
        self.config_manager.load_all_configs()
        
        logger.info(f"智能体工厂初始化完成，可用智能体: {len(self.config_manager.agents_config)}")
    
    def create_agent(self, agent_name: str, force_recreate: bool = False) -> Optional[Union[SmolaAgentsSupervisor, SmolaAgentsMicro]]:
        """创建智能体实例"""
        # 检查缓存
        if not force_recreate and agent_name in self._agent_cache:
            logger.debug(f"从缓存返回智能体: {agent_name}")
            return self._agent_cache[agent_name]
        
        # 获取配置
        config = self.config_manager.get_agent_config(agent_name)
        if not config:
            logger.error(f"智能体配置未找到: {agent_name}")
            return None
        
        if not config.enabled:
            logger.warning(f"智能体已禁用: {agent_name}")
            return None
        
        try:
            # 根据类型创建实例
            if config.type == AgentType.SUPERVISOR:
                agent = SmolaAgentsSupervisor(config, self.config_manager)
            elif config.type == AgentType.MICRO:
                agent = SmolaAgentsMicro(config, self.config_manager)
            else:
                logger.error(f"未知智能体类型: {config.type}")
                return None
            
            # 缓存实例
            self._agent_cache[agent_name] = agent
            
            logger.info(f"成功创建智能体: {agent_name} ({config.type.value})")
            return agent
            
        except Exception as e:
            logger.error(f"创建智能体失败 {agent_name}: {e}")
            return None
    
    def get_or_create_agent(self, agent_name: str) -> Optional[Union[SmolaAgentsSupervisor, SmolaAgentsMicro]]:
        """获取或创建智能体实例"""
        return self.create_agent(agent_name, force_recreate=False)
    
    def list_available_agents(self, agent_type: Optional[AgentType] = None) -> Dict[str, str]:
        """列出可用的智能体"""
        agents = {}
        
        for name, config in self.config_manager.agents_config.items():
            if not config.enabled:
                continue
            if agent_type and config.type != agent_type:
                continue
            
            agents[name] = config.description
        
        return agents
    
    def get_supervisors(self) -> Dict[str, SmolaAgentsSupervisor]:
        """获取所有监督智能体"""
        supervisors = {}
        
        supervisor_names = self.config_manager.list_agents(AgentType.SUPERVISOR)
        for name in supervisor_names:
            agent = self.create_agent(name)
            if isinstance(agent, SmolaAgentsSupervisor):
                supervisors[name] = agent
        
        return supervisors
    
    def get_micro_agents(self) -> Dict[str, SmolaAgentsMicro]:
        """获取所有微智能体"""
        micro_agents = {}
        
        micro_names = self.config_manager.list_agents(AgentType.MICRO)
        for name in micro_names:
            agent = self.create_agent(name)
            if isinstance(agent, SmolaAgentsMicro):
                micro_agents[name] = agent
        
        return micro_agents
    
    def reload_config(self, agent_name: Optional[str] = None):
        """重新加载配置"""
        if agent_name:
            # 重新加载特定智能体配置
            if agent_name in self._agent_cache:
                del self._agent_cache[agent_name]
            logger.info(f"重新加载智能体配置: {agent_name}")
        else:
            # 重新加载所有配置
            self.config_manager.load_all_configs()
            self._agent_cache.clear()
            logger.info("重新加载所有智能体配置")
    
    def create_default_agents(self):
        """创建默认智能体配置和实例"""
        # 创建默认配置文件
        self.config_manager.create_default_config_files()
        
        # 重新加载配置
        self.config_manager.load_all_configs()
        
        # 创建默认智能体实例
        default_agents = ['default_supervisor', 'search_agent', 'analysis_agent']
        
        created_agents = []
        for agent_name in default_agents:
            agent = self.create_agent(agent_name)
            if agent:
                created_agents.append(agent_name)
        
        logger.info(f"创建默认智能体完成: {created_agents}")
        return created_agents
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict]:
        """获取智能体状态"""
        agent = self._agent_cache.get(agent_name)
        if agent:
            return agent.get_status()
        return None
    
    def shutdown_agent(self, agent_name: str):
        """关闭智能体"""
        if agent_name in self._agent_cache:
            del self._agent_cache[agent_name]
            logger.info(f"智能体已关闭: {agent_name}")
    
    def shutdown_all(self):
        """关闭所有智能体"""
        count = len(self._agent_cache)
        self._agent_cache.clear()
        logger.info(f"所有智能体已关闭: {count} 个")
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict]:
        """获取智能体详细信息"""
        config = self.config_manager.get_agent_config(agent_name)
        if not config:
            return None
        
        info = {
            'name': config.name,
            'type': config.type.value,
            'description': config.description,
            'enabled': config.enabled,
            'task_categories': [cat.value for cat in config.task_categories],
            'tools_count': len(config.tools),
            'model': config.model.name,
            'max_steps': config.max_steps,
            'timeout': config.timeout,
            'is_loaded': agent_name in self._agent_cache
        }
        
        if config.type == AgentType.SUPERVISOR:
            info['micro_agents'] = config.micro_agents
            info['sop_category'] = config.sop_category
        
        return info
    
    def __str__(self) -> str:
        cached_count = len(self._agent_cache)
        available_count = len(self.config_manager.agents_config)
        return f"AgentFactory(缓存: {cached_count}, 可用: {available_count})"
    
    def __repr__(self) -> str:
        return f"AgentFactory(config_manager={self.config_manager}, cached_agents={list(self._agent_cache.keys())})"


# 便捷函数
def create_agent_from_config(config_path: Union[str, Path], 
                           config_manager: Optional[ConfigManager] = None) -> Optional[Union[SmolaAgentsSupervisor, SmolaAgentsMicro]]:
    """从配置文件创建智能体"""
    try:
        # 加载配置
        config_manager = config_manager or ConfigManager()
        agent_config = config_manager.load_agent_config(config_path)
        
        if not agent_config:
            logger.error(f"加载配置失败: {config_path}")
            return None
        
        # 创建智能体
        if agent_config.type == AgentType.SUPERVISOR:
            return SmolaAgentsSupervisor(agent_config, config_manager)
        elif agent_config.type == AgentType.MICRO:
            return SmolaAgentsMicro(agent_config, config_manager)
        else:
            logger.error(f"未知智能体类型: {agent_config.type}")
            return None
            
    except Exception as e:
        logger.error(f"从配置创建智能体失败: {e}")
        return None


def create_demo_factory() -> AgentFactory:
    """创建演示工厂，包含默认智能体"""
    factory = AgentFactory()
    
    # 如果没有配置，创建默认配置
    if not factory.config_manager.agents_config:
        factory.create_default_agents()
    
    return factory