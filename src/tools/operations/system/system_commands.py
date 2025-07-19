# -*- coding: utf-8 -*-
"""
Tools for interacting with the system, shell, and Python environment.
"""

import os
import json
import psutil
import subprocess
import sys
from typing import Optional

def get_system_info() -> str:
    """Implementation of the get system info tool."""
    try:
        return json.dumps({
            "success": True,
            "system_info": {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

def execute_shell_command(command: str, working_dir: Optional[str] = None, timeout: int = 30) -> str:
    """Implementation of the execute shell command tool."""
    try:
        dangerous_commands = ['rm -rf', 'del /f', 'format', 'shutdown', 'reboot']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return json.dumps({"error": "Dangerous command execution is forbidden."})
        
        cwd = working_dir if working_dir and os.path.exists(working_dir) else os.getcwd()
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        return json.dumps({
            "success": True,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout} seconds."})
    except Exception as e:
        return json.dumps({"error": str(e)})

def check_python_environment() -> str:
    """Implementation of the check python environment tool."""
    try:
        from importlib import metadata
        
        installed_packages = {dist.metadata['name']: dist.version for dist in metadata.distributions()}
        
        return json.dumps({
            "success": True,
            "python_version": sys.version,
            "python_executable": sys.executable,
            "installed_packages": installed_packages,
            "total_packages": len(installed_packages)
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def install_python_package(package_name: str, version: Optional[str] = None) -> str:
    """Implementation of the install python package tool."""
    try:
        package_spec = f"{package_name}=={version}" if version else package_name
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return json.dumps({"error": f"Package installation failed: {result.stderr}"})
        
        return json.dumps({
            "success": True,
            "output": result.stdout
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}) 