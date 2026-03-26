from langchain_community.tools import ShellTool
import warnings
warnings.filterwarnings('ignore')

def get_py_output(code):
    
    with open("test.py",'w') as file:
        file.write(code)

    shell_tool=ShellTool()

    output=shell_tool.run("python test.py")
    return output