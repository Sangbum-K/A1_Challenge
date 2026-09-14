import argparse
import json

def parse_args(json_path):
    # load json file
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to Namespace
    args = argparse.Namespace(**config_dict)
    return args

if __name__ == "__main__":
    # example
    json_file_path = "config.json"  
    args = parse_args(json_file_path)
    print(args.data_root) 
    print(args.lstm_num_layers)