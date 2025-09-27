import torch
from agent import Agent
from config import YamlParser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':
    config = YamlParser(path='config/dynamic_obstacle.yaml').get_config()
    print(config)
    print('=================================================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
    model_id = 'Turbo'
    agent = Agent(config=config, model_id=model_id, device=device)
    agent.train()
    agent.close()
