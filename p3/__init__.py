import os
import pathlib

# Define the root path of the P3 package
PATH_P3 = str(pathlib.Path(__file__).parent.absolute())

def resolve_path(path_value, path_root):
    """
    Resolve a file path for a configuration parameter.
    - path_value: the value from the config file
    - path_root: the root directory (can be empty)
    Returns the resolved absolute path or '' if empty.
    """
    if not path_value or path_value == '':
        return ''
    if path_root == '' and path_value.lstrip('/').startswith('aoSystem'):
        return os.path.join(PATH_P3, path_value.lstrip('/'))
    return os.path.join(path_root, path_value)