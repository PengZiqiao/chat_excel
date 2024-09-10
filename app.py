import gradio as gr
from gradio_agentchatbot import AgentChatbot, ChatMessage, ThoughtMetadata, ChatFileMessage
from agent import agent_execute, create_agent
from pathlib import Path

def handle_file_upload(file):
    global agent
    
    # 转存
    # TODO: 设置不同的文件名
    data = Path(file).read_bytes()
    Path('data.xlsx').write_bytes(data)

    agent = create_agent('data.xlsx')

def tool_call_message(tool_name, tool_input, observation):
    # 处理工具调用类消息
    content = f"Input\n{tool_input}\nObservation\n```{observation}```"
    return ChatMessage(
        role='assistant', 
        content=content, 
        thought_metadata=ThoughtMetadata(tool_name=tool_name)
    )

def image_message(file_path):
    # 处理图片消息
    f =gr.FileData(path=file_path, mime_type="image/png")
    return ChatFileMessage(
        role='assistant', 
        file=f,
        alt_text="[image]"
    )

def text_message(text):
    # 处理文本消息
    return ChatMessage(
        role='assistant', 
        content=text
    )

def clean_history(history):
    messages = []
    for message in history[:-1]:
        if message.role == 'user':
            human = message.content
            continue # 跳过本轮，等待assistant消息
        elif message.role == 'assistant':
            if message.thought_metadata.tool_name:
                continue # 跳过本轮，等待assistant text消息
            else:
                if isinstance(message, ChatFileMessage):
                    ai = message.alt_text
                elif isinstance(message, ChatMessage):
                    ai = message.content
                else:
                    # 实际应该也不可能发生吧。
                    raise ValueError(f"Unknown message type: {type(message)}")
        messages.append([human, ai])
    
    return messages

def chat(query, history):
    history.append(ChatMessage(role='user', content=query))
    yield "", history

    for chunk in agent_execute(agent, query, clean_history(history)):
        type_ , data = chunk['type'], chunk['data']
        if type_ == 'tool_call':
            message = tool_call_message(data['tool_name'], data['tool_input'], data['observation'])
            history.append(message)
            yield "", history

        elif type_ == 'image':
            message = image_message(data['file_path'])
            history.append(message)
            yield "", history

        elif type_ == 'text':
            message = text_message(data['content'])
            history.append(message)
            yield "", history

with gr.Blocks() as demo:
    file_input = gr.File()
    file_input.upload(handle_file_upload, file_input)

    chatbot = AgentChatbot()
    # chatbot = gr.Chatbot()

    # 如果有文件上传，初始化一个聊天界面；如果没有文件上传，显示提示信息
    @gr.render(inputs=file_input)
    def show_chat(file):
        if file:
            text_input = gr.Textbox(lines=1)
            text_input.submit(chat, [text_input, chatbot], [text_input, chatbot])


if __name__ == "__main__":
    demo.queue().launch()