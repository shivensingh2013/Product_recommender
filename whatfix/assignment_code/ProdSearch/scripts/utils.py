import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Model Configuration")

    # Add arguments that you want to parse, similar to your previous code
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')    
    args = parser.parse_args()
    return args