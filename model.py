
import yaml
from langchain_openai import ChatOpenAI

# 读入配置文件，使用zhipuai glm-4-plus模型
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    config = config['zhipuai glm-4-plus']

# 初始化模型
llm = ChatOpenAI(
    model=config['model'], 
    openai_api_key=config['api_key'],
    openai_api_base=config['base_url'],
    temperature=0.1
)