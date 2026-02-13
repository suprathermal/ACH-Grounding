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

            # Check if value looks like a list (starts with '[' and ends with ']')
            looks_like_list = value_str.strip().startswith('[') and value_str.strip().endswith(']')
            
            # Try to parse the value
            try:
                # First, try to convert to a list if it looks like one
                # The strings may be quoted or not. AI wrote the code to handle this, I would not be able to pick all the details :)
                if looks_like_list:
                    # Remove outer brackets and whitespace
                    list_content = value_str.strip('[] ').strip()
                    # Parse the list content: handle both quoted and unquoted strings
                    value = []
                    if list_content:  # Only parse if there's content
                        # Split on commas while respecting quoted strings
                        # Use a simple state machine approach
                        items = []
                        current_item = []
                        in_quotes = False
                        quote_char = None
                        i = 0
                        while i < len(list_content):
                            char = list_content[i]
                            if not in_quotes:
                                if char in ['"', "'"]:
                                    in_quotes = True
                                    quote_char = char
                                    current_item.append(char)
                                elif char == ',':
                                    # End of current item
                                    item_str = ''.join(current_item).strip()
                                    if item_str:
                                        items.append(item_str)
                                    current_item = []
                                else:
                                    current_item.append(char)
                            else:
                                # Inside quotes
                                current_item.append(char)
                                if char == quote_char:
                                    # Check if it's escaped
                                    if i > 0 and list_content[i-1] == '\\':
                                        # Escaped quote, continue
                                        pass
                                    else:
                                        # End of quoted string
                                        in_quotes = False
                                        quote_char = None
                            i += 1
                        # Add the last item
                        item_str = ''.join(current_item).strip()
                        if item_str:
                            items.append(item_str)
                        
                        # Process each item: remove quotes if present
                        for item in items:
                            item = item.strip()
                            # If it's a quoted string, remove quotes and handle escapes
                            if (item.startswith('"') and item.endswith('"')) or \
                               (item.startswith("'") and item.endswith("'")):
                                # Remove outer quotes
                                quote_char = item[0]
                                inner = item[1:-1]
                                # Handle escaped quotes and backslashes
                                inner = inner.replace(f'\\{quote_char}', quote_char)
                                inner = inner.replace('\\\\', '\\')
                                value.append(inner)
                            else:
                                # Unquoted string, just strip whitespace
                                value.append(item)
                else:
                    value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError) as e:
                # If that fails and it looks like a list, raise an error since we can't parse it
                if looks_like_list:
                    raise ValueError(f"Failed to parse list value at line {line_num}: {value_str}. Error: {e}")
                # If that fails, treat as a string (remove quotes if present)
                # Remove surrounding quotes if they exist
                if (value_str.startswith('"') and value_str.endswith('"')) or \
                   (value_str.startswith("'") and value_str.endswith("'")):
                    value = value_str[1:-1]
                else:
                    value = value_str
            
            config_dict[key] = value
    
    return config_dict

