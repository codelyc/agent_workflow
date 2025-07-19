"""
文件操作工具

提供文件读写、目录操作、文件搜索等基础功能
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def read_file_content(file_path: str, encoding: str = "utf-8") -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        return f"读取文件失败: {e}"


def write_file_content(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """写入文件内容"""
    try:
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return f"文件写入成功: {file_path}"
    except Exception as e:
        logger.error(f"写入文件失败 {file_path}: {e}")
        return f"写入文件失败: {e}"


def append_file_content(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """追加文件内容"""
    try:
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
        return f"内容追加成功: {file_path}"
    except Exception as e:
        logger.error(f"追加文件失败 {file_path}: {e}")
        return f"追加文件失败: {e}"


def list_directory(directory: str, pattern: str = "*") -> str:
    """列出目录内容"""
    try:
        path = Path(directory)
        if not path.exists():
            return f"目录不存在: {directory}"
        
        result = f"目录内容: {directory}\n"
        result += "=" * 50 + "\n"
        
        # 列出文件和目录
        for item in path.glob(pattern):
            if item.is_file():
                size = item.stat().st_size
                result += f"📄 {item.name} ({size} bytes)\n"
            elif item.is_dir():
                result += f"📁 {item.name}/\n"
        
        return result
    except Exception as e:
        logger.error(f"列出目录失败 {directory}: {e}")
        return f"列出目录失败: {e}"


def create_directory(directory: str) -> str:
    """创建目录"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return f"目录创建成功: {directory}"
    except Exception as e:
        logger.error(f"创建目录失败 {directory}: {e}")
        return f"创建目录失败: {e}"


def delete_file(file_path: str) -> str:
    """删除文件"""
    try:
        Path(file_path).unlink()
        return f"文件删除成功: {file_path}"
    except Exception as e:
        logger.error(f"删除文件失败 {file_path}: {e}")
        return f"删除文件失败: {e}"


def copy_file(source: str, destination: str) -> str:
    """复制文件"""
    try:
        shutil.copy2(source, destination)
        return f"文件复制成功: {source} -> {destination}"
    except Exception as e:
        logger.error(f"复制文件失败 {source} -> {destination}: {e}")
        return f"复制文件失败: {e}"


def move_file(source: str, destination: str) -> str:
    """移动文件"""
    try:
        shutil.move(source, destination)
        return f"文件移动成功: {source} -> {destination}"
    except Exception as e:
        logger.error(f"移动文件失败 {source} -> {destination}: {e}")
        return f"移动文件失败: {e}"


def search_files(directory: str, pattern: str = "*.py", max_results: int = 50) -> str:
    """搜索文件"""
    try:
        path = Path(directory)
        if not path.exists():
            return f"目录不存在: {directory}"
        
        files = list(path.rglob(pattern))
        files = files[:max_results]  # 限制结果数量
        
        result = f"文件搜索结果: {directory}\n"
        result += f"模式: {pattern}\n"
        result += f"找到 {len(files)} 个文件\n"
        result += "=" * 50 + "\n"
        
        for file_path in files:
            rel_path = file_path.relative_to(path)
            size = file_path.stat().st_size
            result += f"📄 {rel_path} ({size} bytes)\n"
        
        return result
    except Exception as e:
        logger.error(f"搜索文件失败 {directory}: {e}")
        return f"搜索文件失败: {e}"


def get_file_info(file_path: str) -> str:
    """获取文件信息"""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"文件不存在: {file_path}"
        
        stat = path.stat()
        
        result = f"文件信息: {file_path}\n"
        result += "=" * 50 + "\n"
        result += f"大小: {stat.st_size} bytes\n"
        result += f"修改时间: {stat.st_mtime}\n"
        result += f"权限: {oct(stat.st_mode)}\n"
        
        if path.is_file():
            result += f"类型: 文件\n"
            # 尝试读取前几行
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:5]
                    result += f"前5行:\n"
                    for i, line in enumerate(lines, 1):
                        result += f"  {i}: {line.rstrip()}\n"
            except:
                result += "无法读取文件内容\n"
        else:
            result += f"类型: 目录\n"
        
        return result
    except Exception as e:
        logger.error(f"获取文件信息失败 {file_path}: {e}")
        return f"获取文件信息失败: {e}"


def find_files_by_extension(directory: str, extension: str) -> str:
    """按扩展名查找文件"""
    return search_files(directory, f"*{extension}") 