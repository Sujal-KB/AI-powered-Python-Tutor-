import os
import warnings
warnings.filterwarnings('ignore')

from langchain_community.tools import ShellTool


def get_py_output(code):

    with open("test.py",'w') as file:
        file.write(code)
    
    file_name="test.py"
    shell_tool = ShellTool()
    output = shell_tool.run(f"python {file_name}")
    return output

    
