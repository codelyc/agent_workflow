"""
Advanced Code Analysis Tools
Powered by Tree-sitter for high-accuracy, multi-language parsing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import tree_sitter
from tree_sitter import Node
from tree_sitter_languages import get_parser, get_language

TREESITTER_AVAILABLE = True

logger = logging.getLogger(__name__)

# Language configuration mapping file extensions to tree-sitter language names
LANGUAGE_MAPPING = {
    ".py": "python",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

# Queries for extracting elements from different languages
QUERIES = {
    "python": {
        "function": "(function_definition name: (identifier) @name) @element",
        "async_function": "(function_definition name: (identifier) @name) @element",
        "class": "(class_definition name: (identifier) @name) @element",
        "method": "(class_definition body: (block . (function_definition name: (identifier) @name)) @element)",
        "async_method": "(class_definition body: (block . (function_definition name: (identifier) @name)) @element)",
    },
    "go": {
        "function": "(function_declaration name: (identifier) @name) @element",
        "method": "(method_declaration name: (field_identifier) @name) @element",
        "struct": '(type_spec name: (type_identifier) @name type: (struct_type)) @element',
        "interface": '(type_spec name: (type_identifier) @name type: (interface_type)) @element',
    },
    "java": {
        "class": "(class_declaration name: (identifier) @name) @element",
        "interface": "(interface_declaration name: (identifier) @name) @element",
        "method": "(method_declaration name: (identifier) @name) @element",
    },
    "javascript": {
        "function": ["(function_declaration name: (identifier) @name) @element", "(arrow_function) @element"],
        "async_function": ["(function_declaration name: (identifier) @name) @element", "(arrow_function) @element"],
        "class": "(class_declaration name: (identifier) @name) @element",
        "method": "(method_definition name: (property_identifier) @name) @element",
        "interface": [],
        "type": [],
    },
    "typescript": {
        "function": ["(function_declaration name: (identifier) @name) @element", "(arrow_function) @element"],
        "async_function": ["(function_declaration name: (identifier) @name) @element", "(arrow_function) @element"],
        "class": "(class_declaration name: (identifier) @name) @element",
        "method": "(method_definition name: (property_identifier) @name) @element",
        "interface": "(interface_declaration name: (type_identifier) @name) @element",
        "type": "(type_alias_declaration name: (type_identifier) @name) @element",
    },
    "c": {
        "function": "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @element",
        "struct": "(struct_specifier name: (type_identifier) @name) @element",
    },
    "cpp": {
        "function": "(function_definition declarator: (function_declarator declarator: (qualified_identifier (identifier) @name))) @element",
        "struct": "(struct_specifier name: (type_identifier) @name) @element",
        "class": "(class_specifier name: (type_identifier) @name) @element",
        "namespace": "(namespace_definition name: (identifier) @name) @element",
    }
}

def _detect_language(path: Path) -> Optional[str]:
    """Detect language from file extension."""
    return LANGUAGE_MAPPING.get(path.suffix.lower())

def _get_docstring(node: Node, language: str) -> str: # type: ignore
    """Extract docstring or leading comments."""
    if not TREESITTER_AVAILABLE: return ""
    if language == "python":
        body = node.child_by_field_name("body") # type: ignore
        if body and body.children and body.children[0].type == "expression_statement":
            string_node = body.children[0].children[0]
            if string_node and string_node.type == "string":
                return string_node.text.decode('utf-8')
    prev_sibling = node.prev_sibling # type: ignore
    if prev_sibling and prev_sibling.type == "comment":
        return prev_sibling.text.decode('utf-8')
    return ""

def _parse_with_treesitter(source_code: str, language: str, element_types: Optional[List[str]], include_docstrings: bool, include_line_numbers: bool) -> List[Dict[str, Any]]:
    """Core Tree-sitter parsing logic."""
    lang_obj = get_language(language)
    parser = get_parser(language)
    tree = parser.parse(bytes(source_code, "utf8"))
    
    elements = []
    
    target_queries = QUERIES.get(language, {})
    types_to_run = element_types if element_types is not None else list(target_queries.keys())
    
    for element_type in types_to_run:
        query_strings = target_queries.get(element_type, [])
        if not isinstance(query_strings, list):
            query_strings = [query_strings]
            
        for query_string in query_strings:
            if not query_string: continue
            try:
                query = lang_obj.query(query_string)
                captures = query.captures(tree.root_node)
            except (tree_sitter.TreeSitterError, Exception): # type: ignore
                continue

            for node, capture_name in captures:
                if not isinstance(node, Node): continue # type: ignore
                element_data: Dict[str, Any] = {}
                if capture_name == "name":
                    continue

                name_node = next((n for n, c in captures if isinstance(n, Node) and c == "name" and n.parent == node), None) # type: ignore
                
                element_data["type"] = element_type
                element_data["name"] = name_node.text.decode('utf-8') if name_node else "anonymous" # type: ignore
                
                if include_line_numbers:
                    element_data["start_line"] = node.start_point[0] + 1 # type: ignore
                    element_data["end_line"] = node.end_point[0] + 1 # type: ignore

                if include_docstrings:
                    element_data["docstring"] = _get_docstring(node, language)
                
                element_data["content"] = node.text.decode('utf-8') # type: ignore
                elements.append(element_data)

    return sorted(elements, key=lambda x: x.get("start_line", 0))

def _format_treesitter_element(element: Dict[str, Any]) -> str:
    """Format a single parsed element into a readable string."""
    line_info = f" (Lines: {element['start_line']}-{element['end_line']})" if 'start_line' in element else ""
    header = f"--- {element['type'].upper()}: {element['name']}{line_info} ---"
    
    docstring = ""
    if element.get("docstring"):
        docstring = f"  \"\"\"\n  {element['docstring'].strip()}\n  \"\"\"\n"

    content = element.get('content', '').strip()
    
    return f"{header}\n{docstring}{content}\n"

def parse_code_with_treesitter(
    file_path: str,
    element_types: Optional[List[str]] = None,
    include_docstrings: bool = True,
    include_line_numbers: bool = True,
    language: Optional[str] = None
) -> str:
    """
    Parse source code using Tree-sitter for accurate multi-language analysis.

    This tool uses Tree-sitter parsers to extract code elements with high accuracy.
    Requires tree-sitter and language-specific packages to be installed.

    Args:
        file_path: Path to the source file to analyze
        element_types: List of element types to extract. If None, extracts all supported types.
                      Supported types by language:

                      Python (.py):
                      - "function": Regular functions (def name(...))
                      - "async_function": Async functions (async def name(...))
                      - "class": Class definitions (class Name(...))
                      - "method": Class methods (functions inside classes)
                      - "async_method": Async class methods

                      Go (.go):
                      - "function": Functions (func name(...))
                      - "method": Methods with receivers (func (r Type) name(...))
                      - "struct": Struct types (type Name struct {...})
                      - "interface": Interface types (type Name interface {...})

                      Java (.java):
                      - "class": Class definitions (class Name {...})
                      - "interface": Interface definitions (interface Name {...})
                      - "method": Methods inside classes/interfaces

                      JavaScript/TypeScript (.js, .ts, .jsx, .tsx):
                      - "function": Regular functions (function name(...))
                      - "async_function": Async functions (async function name(...))
                      - "class": ES6 classes (class Name {...})
                      - "method": Class methods
                      - "interface": TypeScript interfaces (interface Name {...})
                      - "type": TypeScript type aliases (type Name = ...)

                      C/C++ (.c, .cpp, .h, .hpp):
                      - "function": Functions (return_type name(...))
                      - "struct": Struct definitions (struct name {...})
                      - "class": C++ classes (class name {...})
                      - "namespace": C++ namespaces (namespace name {...})

        include_docstrings: Whether to include documentation/comments
        include_line_numbers: Whether to include line number information
        language: Force specific language detection (auto-detected from file extension if None)

    Returns:
        str: Detailed analysis of code elements using Tree-sitter with high accuracy

    Examples:
        >>> parse_code_with_treesitter("src/utils.py", ["function", "class"])
        >>> parse_code_with_treesitter("main.go", ["function", "struct"])
        >>> parse_code_with_treesitter("app.ts", ["function", "class", "interface"])
        >>> parse_code_with_treesitter("service.java", ["class", "method"])
    """
    if not TREESITTER_AVAILABLE:
        return "❌ Tree-sitter not available. Please install: pip install tree-sitter tree-sitter-languages"

    if not file_path or not file_path.strip():
        raise ValueError("file_path is required and cannot be empty")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist")

    detected_language = language or _detect_language(path)

    if not detected_language:
        return f"❌ Could not detect language for file '{file_path}'.\n✅ Supported extensions: {', '.join(LANGUAGE_MAPPING.keys())}"
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        elements = _parse_with_treesitter(
            source_code, detected_language, element_types,
            include_docstrings, include_line_numbers
        )

        result = f"Tree-sitter Analysis for: {file_path} ({detected_language.upper()})\n"
        result += "=" * 70 + "\n"

        ts_version = 'unknown'
        if TREESITTER_AVAILABLE and hasattr(tree_sitter, '__version__'):
            ts_version = tree_sitter.__version__ # type: ignore

        result += f"Parser: Tree-sitter v{ts_version}\n"
        result += f"Language: {detected_language}\n"
        result += f"Elements found: {len(elements)}\n"
        result += "=" * 70 + "\n\n"

        if not elements:
            searched_for = ', '.join(element_types) if element_types else 'all supported types'
            result += f"No code elements found matching the specified criteria.\nSearched for: {searched_for}\n"
            return result

        for i, element in enumerate(elements, 1):
            result += f"[{i}/{len(elements)}] "
            result += _format_treesitter_element(element) + "\n"

        result += f"Successfully parsed {len(elements)} elements using Tree-sitter\n"
        return result

    except Exception as e:
        error_msg = f"Error parsing {detected_language} file '{file_path}' with Tree-sitter: {e}\n"
        error_msg += "Please ensure the language parser is correctly installed via 'tree-sitter-languages'."
        return error_msg 