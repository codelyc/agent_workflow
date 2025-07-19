# -*- coding: utf-8 -*-
"""
Tools for interacting with Git repositories.
"""

import os
import json
import subprocess

def git_status(repo_path: str = ".") -> str:
    """Implementation of the git status tool."""
    try:
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return json.dumps({"error": f"Not a Git repository: {repo_path}"})
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            return json.dumps({"error": f"Git command failed: {result.stderr}"})
        
        status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        modified = [line[3:] for line in status_lines if line.startswith(' M') or line.startswith('M ')]
        untracked = [line[3:] for line in status_lines if line.startswith('??')]
        staged = [line[3:] for line in status_lines if line.startswith('A ')]
        
        return json.dumps({
            "success": True,
            "modified_files": modified,
            "untracked_files": untracked,
            "staged_files": staged,
            "has_changes": bool(status_lines)
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def git_commit(message: str, repo_path: str = ".") -> str:
    """Implementation of the git commit tool."""
    try:
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return json.dumps({"error": f"Not a Git repository: {repo_path}"})
        
        add_result = subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, text=True, timeout=30)
        if add_result.returncode != 0:
            return json.dumps({"error": f"Git add failed: {add_result.stderr}"})
        
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        
        if commit_result.returncode != 0:
            if "nothing to commit" in commit_result.stdout:
                 return json.dumps({"success": True, "message": "No changes to commit."})
            return json.dumps({"error": f"Git commit failed: {commit_result.stderr}", "output": commit_result.stdout})
        
        return json.dumps({"success": True, "output": commit_result.stdout})
        
    except Exception as e:
        return json.dumps({"error": str(e)}) 