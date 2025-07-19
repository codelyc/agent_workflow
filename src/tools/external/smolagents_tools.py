# -*- coding: utf-8 -*-
"""
Smol Gpt Agent Tools Registration Hub

This module dynamically loads and registers tools for the agent framework
based on the settings in `configs/tools.yaml`.
"""
import inspect
from smolagents import tool
from ...core.config.config_parser import load_tools_config
from ..tool_descriptions import TOOL_DESCRIPTIONS

# Import all possible tool implementations
from ..development.code_analysis.treesitter_parser import parse_code_with_treesitter
from ..analysis.text.text_grep import search_keyword_in_directory, search_files
from ..operations.system.system_commands import get_system_info, execute_shell_command, check_python_environment, install_python_package
from ..development.git.git_commands import git_status, git_commit
from ..operations.system.docker_commands import docker_list_containers, docker_run_container
from ..development.memory_commands import save_memory, retrieve_memory
from ..development.file_ops.file_operations import read_file_content, write_file_content
from ..development.file_ops.project_analysis import analyze_project_structure

# A registry of all available tool implementations
ALL_TOOLS = {
    "parse_code_file_with_treesitter": parse_code_with_treesitter,
    "search_keyword_in_directory": search_keyword_in_directory,
    "search_files": search_files,
    "get_system_info": get_system_info,
    "execute_shell_command": execute_shell_command,
    "check_python_environment": check_python_environment,
    "install_python_package": install_python_package,
    "git_status": git_status,
    "git_commit": git_commit,
    "docker_list_containers": docker_list_containers,
    "docker_run_container": docker_run_container,
    "save_memory": save_memory,
    "retrieve_memory": retrieve_memory,
    "read_file_content": read_file_content,
    "write_file_content": write_file_content,
    "analyze_project_structure": analyze_project_structure,
}

# Load the enabled tools from the YAML configuration
ENABLED_TOOLS = load_tools_config()

# Dynamically create and register the enabled tools in the global scope of this module
for tool_name in ENABLED_TOOLS:
    if tool_name in ALL_TOOLS:
        tool_func = ALL_TOOLS[tool_name]
        
        # Assign the detailed docstring
        tool_func.__doc__ = TOOL_DESCRIPTIONS.get(tool_name, tool_func.__doc__)
        
        # Apply the decorator and assign to the global scope
        globals()[tool_name] = tool(tool_func)

def list_available_tools():
    """Returns a list of the currently enabled tools."""
    return ENABLED_TOOLS 