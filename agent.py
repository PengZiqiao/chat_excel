from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from functions import create_tool, extract_python_code, parse_tool_input, remove_image_markdown
from model import llm


def create_prompt_template(dhead):
    system_prompt = load_prompt("dataminer.yaml")

    return ChatPromptTemplate([
        ("system", system_prompt.format(dhead=dhead)), 
        ("placeholder", "{messages}"),
        ("system", "根据以上对话，编写python代码，选择工具执行。按执行结果回复用户。"),
        ("placeholder", "{agent_scratchpad}")
    ])


def create_agent(file_path):
    tool, dhead = create_tool(file_path)
    prompt_template = create_prompt_template(dhead)

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[tool],
        prompt=prompt_template
    )

    return AgentExecutor(agent=agent, tools=[tool])


def agent_execute(agent, query, history):
    # 组装messages
    messages = []
    for human_message, ai_message in history:
        messages.append(HumanMessage(content=human_message))
        messages.append(AIMessage(content=ai_message))
    messages.append(HumanMessage(content=query))

    # 调用agent
    for chunk in agent.stream({'messages': messages}):
        # 执行工具
        if 'actions' in chunk.keys():
            actions = chunk['actions'][0]
            tool_name = actions.tool
            tool_input = parse_tool_input(actions.tool_input)
            # 不输出，等待下一个消息一并输出结果
        # 工具结果
        elif 'steps' in chunk.keys():
            steps = chunk['steps'][0]
            observation = steps.observation

            # 连同工具输入一起返回
            yield {
                'type': 'tool_call',
                'data': {
                    'tool_name': tool_name,
                    'tool_input': tool_input,
                    'observation': observation,
                }
            }

            # 如果结果是保存了图片，再额外返回图片地址
            if observation == '图片已生成，请提示用户查看。':
                file_path = 'plot.png'
                yield {
                    'type': 'image',
                    'data': {
                        'file_path': file_path
                    }
                }

        # 普通消息
        else:
            output = chunk['output']

            # 如果回复中有python代码，则手动执行python代码
            python_code = extract_python_code(output)
            if python_code:
                content = agent.tools[0].invoke(python_code)
                yield {
                    'type': 'tool_call',
                    'data': {
                        'tool_name': 'manual_python_repl',
                        'tool_input': f"```python\n{python_code}\n```",
                        'observation': content
                    }
                }
                
                # 如果结果是保存了图片，再额外返回图片地址
                if observation == '图片已生成，请提示用户查看。':
                    file_path = 'plot.png'
                    yield {
                        'type': 'image',
                        'data': {
                            'file_path': file_path
                        }
                    }

            else:
                # 普通文本消息，将消息中表示图片的markdown去掉，反正又不能正常显示
                content = remove_image_markdown(output)
                yield {
                    'type': 'text',
                    'data': {
                        'content': content
                    }
                }
