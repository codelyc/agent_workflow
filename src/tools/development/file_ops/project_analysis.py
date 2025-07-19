# -*- coding: utf-8 -*-
"""
Tools for analyzing project structure.
"""

import os
import json
from pathlib import Path

def analyze_project_structure(project_path: str) -> str:
    """Implementation of the analyze project structure tool."""
    try:
        if not os.path.exists(project_path):
            return json.dumps({"error": f"Project path does not exist: {project_path}"})
        
        file_stats = {}
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(project_path):
            # Ignore common virtual environment and cache directories
            dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', 'node_modules', '.git']]
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    ext = os.path.splitext(file)[1].lower() or ".no_extension"
                    file_stats[ext] = file_stats.get(ext, 0) + 1
                    total_files += 1
                    total_size += size
                except OSError:
                    continue
        
        analysis = f"Project Structure Analysis for: {project_path}\n"
        analysis += f"Total Files: {total_files}\n"
        analysis += f"Total Size: {total_size / (1024*1024):.2f} MB\n\n"
        analysis += "File Type Distribution:\n"
        
        sorted_stats = sorted(file_stats.items(), key=lambda item: item[1], reverse=True)
        
        for ext, count in sorted_stats:
            percentage = (count / total_files * 100) if total_files > 0 else 0
            analysis += f"  - {ext}: {count} files ({percentage:.1f}%)\n"
        
        return json.dumps({"success": True, "analysis": analysis}, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)}) 