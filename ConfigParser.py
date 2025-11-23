import ast
import os
import re
from typing import Dict, Any

def parse_config(config_path: str) -> Dict[str, Any]:
    """
    Reads a config file from a given path and returns a key-value dictionary.
    
    Config format: Each line should be formatted as 'key = value'
    Values can be:
    - Strings (quoted or unquoted)
    - Booleans (True, False)
    - Numbers (int, float)
    - Lists (Python format, e.g., ["foo", "bar"])
    - Dictionaries (Python format, e.g., {"key": "value"})
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Dictionary with keys and parsed values from the config file
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If a line cannot be parsed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config_dict = {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # Strip whitespace and skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split on '=' to separate key and value
            if '=' not in line:
                raise ValueError(f"Invalid config format at line {line_num}: missing '=' separator")
            
            parts = line.split('=', 1)  # Split only on first '='
            key = parts[0].strip()
            value_str = parts[1].strip()
            
            if not key:
                raise ValueError(f"Empty key at line {line_num}")
            
            # Preprocess dictionary strings to handle blank values after colons
            # Replace patterns like ": ," or ": }" with ": None," or ": None}"
            if value_str.strip().startswith('{') and value_str.strip().endswith('}'):
                # Use regex to find colons followed by whitespace and then comma or closing brace
                # Pattern matches: ":" followed by optional whitespace, then comma or closing brace
                value_str = re.sub(r':\s*(?=,|})', ': None', value_str)
            
            # Try to parse the value
            try:
                # First, try to evaluate as Python literal (for lists, dicts, bools, numbers)
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # If that fails, treat as a string (remove quotes if present)
                # Remove surrounding quotes if they exist
                if (value_str.startswith('"') and value_str.endswith('"')) or \
                   (value_str.startswith("'") and value_str.endswith("'")):
                    value = value_str[1:-1]
                else:
                    value = value_str
            
            config_dict[key] = value
    
    return config_dict

