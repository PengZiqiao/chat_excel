from langchain_core.tools import tool
from pbox import CodeSandBox
import re
from typing import Annotated


def create_tool(file_path):
    sandbox, dhead = create_code_sandbox(file_path)

    @tool
    def python_repl(
        code: Annotated[str, (
                "A valid python command working with `df`. ",
                "If you want to see the output of a value, you should print it out with `print(...)`. ",
                "Show the plot with `plt.show()`. "
            )]
    ) -> str:
        """
        A Python shell. Use this to execute python commands with `df` as the dataframe.
        """
        return code_execute(code, sandbox)
    
    return python_repl, dhead

def create_code_sandbox(file_path):
    sandbox = CodeSandBox()
    code = '\n'.join((
        "import matplotlib.pyplot as plt",
        "import matplotlib as mpl",
        "import pandas as pd",
        "mpl.font_manager.fontManager.addfont('simhei.ttf')",
        "mpl.rc('font', family='SimHei')",
        "plt.rcParams['axes.unicode_minus']=False",
        f"df = pd.read_excel('{file_path}')",
        "print(df.head().to_markdown())"
    ))

    # TODO: add try catch，对传入file_path不能识别的情况进行处理
    result = sandbox.execute_code(code)

    dhead = '\n'.join(result.logs.stdout)

    return sandbox, dhead

def parse_tool_input(tool_input):
    raw_text = tool_input['code']
    code = '\n'.join((
        "```python",
        raw_text.replace(r'\n', '<br/>'),
        "```"
    ))

    return code

def code_execute(code, sandbox):
    result = sandbox.execute_code(code)

    # 正常获得`print(...)`的输出
    if result.logs.stdout:
        output = '\n'.join(result.logs.stdout)
        return f"已得到执行结果：\n{output}"

    # AI没有使用`print(...)`，只执行了统计或使用了`plt.show()`
    elif result.results:
        image_data = next(
            (d['data'] for d in result.results if d['type'] == 'image/png'),
            None
        )

        # 如何有图片，就保存到本地
        # TODO: 控制输出文件名，避免多人使用时，文件名冲突
        if image_data:
            save_plot(image_data)
            return "图片已生成，请提示用户查看。"
        # 尝试获得text/plain输出
        else:
            text_data = next(
                (d['data'] for d in result.results if d['type'] == 'text/plain'),
                None
            )
            return f"已得到执行结果：\n{text_data}" if text_data else f"未获得执行结果，请检查代码是否正确。"

    # 执行出错，打印error traceback 信息
    elif result.error:
        # 用正则匹配ansi颜色代码后，替换为空
        pattern = r'\x1b\[\d+[;\d+]*m'
        error_message =  '\n'.join(result.error.traceback)
        output = re.sub(pattern, '', error_message)

        return f"执行出错：\n{output}"
    # 其他情况，直接返回整个result
    else:
        return str(result.json())
    
def save_plot(data, file_path='plot.png'):
    from io import BytesIO  
    from PIL import Image  
    import base64

    image_data = data
    image_data_bytes = base64.b64decode(image_data)  
    image = Image.open(BytesIO(image_data_bytes))
    image.save(file_path, format='PNG')

def extract_python_code(text):
    import re
    pattern = r"```(?:python)?([\s\S]*?)```"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 返回匹配的第一个组（括号中的内容）
        return match.group(1)
    else:
        # 如果没有匹配到，返回None
        return None
    

def remove_image_markdown(text):
    import re
    pattern = r"!\[.*?\]\(.*?\)"
    return re.sub(pattern, "", text, re.MULTILINE)