# -*- coding: utf-8 -*-
"""
Tools for interacting with Docker containers.
"""

import json
import subprocess
from typing import Optional

def docker_list_containers() -> str:
    """Implementation of the docker list containers tool."""
    try:
        docker_check = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if docker_check.returncode != 0:
            return json.dumps({"error": "Docker is not available. Please ensure it is installed and running."})
        
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            return json.dumps({"error": f"Docker command failed: {result.stderr}"})
        
        containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
        
        return json.dumps({
            "success": True,
            "containers": containers,
            "total_containers": len(containers)
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def docker_run_container(image: str, command: Optional[str] = None, ports: Optional[str] = None) -> str:
    """Implementation of the docker run container tool."""
    try:
        docker_check = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if docker_check.returncode != 0:
            return json.dumps({"error": "Docker is not available. Please ensure it is installed and running."})
        
        docker_cmd = ["docker", "run", "-d"]
        if ports:
            docker_cmd.extend(["-p", ports])
        
        docker_cmd.append(image)
        
        if command:
            docker_cmd.extend(command.split())
        
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            return json.dumps({"error": f"Docker run failed: {result.stderr}"})
        
        container_id = result.stdout.strip()
        
        return json.dumps({
            "success": True,
            "container_id": container_id,
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)}) 