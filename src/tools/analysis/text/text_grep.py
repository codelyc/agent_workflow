# -*- coding: utf-8 -*-
"""
Tools for searching files and keywords in the workspace.
"""

import os
import json
import glob
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

def _find_best_searcher() -> Optional[str]:
    """Find the best available command-line search tool."""
    for tool_name in ["ag", "rg", "grep"]:
        if shutil.which(tool_name):
            return tool_name
    return None

def search_keyword_in_directory(
    directory: str,
    keyword: str,
    max_output_lines: int = 100,
    case_sensitive: bool = False,
    include_line_numbers: bool = True,
    file_extensions: Optional[List[str]] = None
) -> str:
    """Implementation of the keyword search tool."""
    searcher = _find_best_searcher()
    if not searcher:
        return json.dumps({"error": "No search tool (ag, rg, grep) found in system PATH."})

    command = [searcher]
    if not case_sensitive:
        command.append("-i")
    if include_line_numbers:
        command.append("-n")
    
    if searcher == "grep":
        command.extend(["-r", keyword, directory])
    else: # ag or rg
        command.extend([keyword, directory])

    if file_extensions:
        for ext in file_extensions:
            clean_ext = ext.lstrip('.')
            if searcher == "rg":
                command.extend(["-g", f"*.{clean_ext}"])
            elif searcher == "ag":
                command.append(f"--{clean_ext}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        output = result.stdout.strip().split('\n')
        if max_output_lines and len(output) > max_output_lines:
            output = output[:max_output_lines]
        
        return json.dumps({
            "success": True, 
            "results": "\n".join(output), 
            "lines_found": len(output),
            "tool_used": searcher
        }, ensure_ascii=False)
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"Search command failed: {e.stderr}", "output": e.stderr})
    except Exception as e:
        return json.dumps({"error": str(e)})

def search_files(directory: str, pattern: str = "*.py", max_results: int = 50) -> str:
    """Implementation of the file search tool."""
    try:
        search_path = Path(directory)
        if not search_path.exists():
            return json.dumps({"error": f"Directory does not exist: {directory}"})
        
        search_pattern = str(search_path / "**" / pattern)
        files = glob.glob(search_pattern, recursive=True)
        
        files = files[:max_results]
        
        results = []
        for file_path in files:
            try:
                file_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                }
                results.append(file_info)
            except OSError:
                continue # Ignore files that might be gone
        
        return json.dumps({
            "success": True,
            "total_found": len(results),
            "files": results
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)}) 