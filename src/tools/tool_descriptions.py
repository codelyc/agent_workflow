# -*- coding: utf-8 -*-
"""
Centralized dictionary of tool descriptions for Smol Gpt agents.
This allows for easy management and decoupling of tool documentation from implementation.
"""

TOOL_DESCRIPTIONS = {
    "search_keyword_in_directory": """
    Search for a keyword in a directory using the best available search tool.

    This function tries to use search tools in the following order of preference:
    1. ag (The Silver Searcher) - fastest and most feature-rich
    2. rg (ripgrep) - very fast and modern
    3. grep - standard Unix tool, available everywhere

    Args:
        directory (str): **REQUIRED** The directory path to search in
        keyword (str): **REQUIRED** The keyword/pattern to search for.
                      **SUPPORTS REGULAR EXPRESSIONS** - you can use regex patterns like:
                      - 'function.*name' for pattern matching
                      - '^import' for lines starting with 'import'
                      - 'TODO|FIXME' for multiple keywords
                      - '\\bclass\\b' for word boundaries
        max_output_lines (int): Maximum number of output lines to return (default: 100, None for unlimited)
        case_sensitive (bool): Whether the search should be case-sensitive (default: False)
        include_line_numbers (bool): Whether to include line numbers in output (default: True)
        file_extensions (Optional[List[str]]): Optional list of file extensions to search (e.g., ['.py', '.js', '.go'])

    Returns:
        str: Search results containing matched lines with file paths and line numbers, in JSON format.
    """,
    "parse_code_file_with_treesitter": """
    Parse source code using Tree-sitter for accurate multi-language analysis.

    This tool uses Tree-sitter parsers to extract code elements with high accuracy, supporting multiple languages.
    
    Args:
        file_path (str): **REQUIRED** Path to the source file to analyze.
        element_types (Optional[List[str]]): List of element types to extract (e.g., ["function", "class"]). 
                                             If None, extracts all supported types for the detected language.
        include_docstrings (bool): Whether to include documentation/comments in the output (default: True).
        include_line_numbers (bool): Whether to include line number information (default: True).
        language (Optional[str]): Force a specific language (e.g., "python"). If None, it's auto-detected from the file extension.

    Returns:
        str: A JSON formatted string containing a detailed analysis of the code elements.
    """,
    "search_files": """
    Find files in a directory by a given glob pattern.

    Args:
        directory (str): **REQUIRED** The directory path to start the search from.
        pattern (str): The glob pattern to match files (e.g., "*.py", "**/*.js"). Default is "*.py".
        max_results (int): The maximum number of file paths to return (default: 50).

    Returns:
        str: A JSON formatted string containing a list of found files with their metadata.
    """,
    "get_system_info": """
    Get system information, including CPU, memory, and disk usage.
    
    Returns:
        str: A JSON formatted string with detailed system metrics.
    """,
    "execute_shell_command": """
    Execute a shell command and return its output, standard error, and exit code.
    
    Note: Dangerous commands like 'rm -rf' are blocked for safety.
    
    Args:
        command (str): **REQUIRED** The shell command to execute.
        working_dir (Optional[str]): The directory to run the command in. Defaults to the current working directory.
        timeout (int): The timeout in seconds for the command execution (default: 30).

    Returns:
        str: A JSON formatted string with the command's stdout, stderr, and return code.
    """,
    "read_file_content": """
    Read the content of a specified file.

    Args:
        file_path (str): **REQUIRED** The path to the file to read.
        max_lines (int): The maximum number of lines to read from the file (default: 1000).

    Returns:
        str: A JSON formatted string containing the file content.
    """,
    "write_file_content": """
    Write content to a specified file, with options to overwrite or append.

    Args:
        file_path (str): **REQUIRED** The path to the file to write to.
        content (str): **REQUIRED** The content to write to the file.
        mode (str): The write mode. "w" to overwrite (default), or "a" to append.

    Returns:
        str: A JSON formatted string confirming the success of the write operation.
    """,
    "analyze_project_structure": """
    Analyze the file structure of a project, showing file types, counts, and total size.

    Args:
        project_path (str): **REQUIRED** The root directory of the project to analyze.

    Returns:
        str: A JSON formatted string containing a summary of the project's structure.
    """,
    "git_status": """
    Get the status of a Git repository.

    Shows modified, untracked, and staged files.

    Args:
        repo_path (str): The path to the Git repository (default: current directory).

    Returns:
        str: A JSON formatted string detailing the repository status.
    """,
    "git_commit": """
    Stage all changes and commit them to a Git repository with a given message.

    Args:
        message (str): **REQUIRED** The commit message.
        repo_path (str): The path to the Git repository (default: current directory).

    Returns:
        str: A JSON formatted string with the output from the git commit command.
    """,
    "docker_list_containers": """
    List all Docker containers, including running and stopped ones.
    
    Returns:
        str: A JSON formatted string containing a list of all Docker containers and their details.
    """,
    "docker_run_container": """
    Run a command in a new, detached Docker container from a specified image.

    Args:
        image (str): **REQUIRED** The Docker image to use (e.g., "ubuntu:latest").
        command (Optional[str]): The command to run inside the container.
        ports (Optional[str]): Port mapping in "host:container" format (e.g., "8080:80").

    Returns:
        str: A JSON formatted string with the ID of the newly created container.
    """,
    "check_python_environment": """
    Check the current Python environment, including version, executable path, and installed packages.

    Returns:
        str: A JSON formatted string with details about the Python environment.
    """,
    "install_python_package": """
    Install a Python package using pip.

    Args:
        package_name (str): **REQUIRED** The name of the package to install.
        version (Optional[str]): The specific version of the package to install. If None, the latest version is installed.

    Returns:
        str: A JSON formatted string with the output from the pip install command.
    """,
    "save_memory": """
    Save a key-value pair to the agent's long-term memory for later retrieval.

    Args:
        key (str): **REQUIRED** The key to store the information under.
        value (str): **REQUIRED** The value to store.
        category (str): An optional category for the memory (default: "general").

    Returns:
        str: A JSON formatted string confirming that the memory was saved.
    """,
    "retrieve_memory": """
    Retrieve a value from the agent's long-term memory using a key.

    Args:
        key (str): **REQUIRED** The key of the memory to retrieve.
        category (str): The category of the memory (default: "general").

    Returns:
        str: A JSON formatted string containing the retrieved value and metadata.
    """,
    "create_workflow_report": """
    Create a JSON report for a completed workflow and save it to a file in the 'reports/' directory.

    Args:
        workflow_name (str): **REQUIRED** The name of the workflow for the report.
        results (str): **REQUIRED** The results or output of the workflow to be included in the report.
        status (str): The final status of the workflow (e.g., "completed", "failed"). Default is "completed".

    Returns:
        str: A JSON formatted string confirming the report creation and the path to the report file.
    """
} 