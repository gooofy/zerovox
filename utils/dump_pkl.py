#!/bin/env python3

import torch
import argparse

def print_tensor_names(model):

    if isinstance(model, dict) and 'state_dict' in model:
        state_dict = model['state_dict']
        for name in state_dict:
            print(name)
    elif isinstance(model, dict) and 'model' in model:

        model = model['model']

        if isinstance(model, dict):
            for k in model:
                print (k)
                print ("=========================")
                print_tensor_names(model[k])
        else:
            state_dict = model['model'].state_dict()
            for name in state_dict:
                print(name)
    elif isinstance(model, dict):
        for name in model:
            print(name)
    else:
        for name in model.state_dict():
            print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print tensor names from a PyTorch checkpoint.")
    parser.add_argument("checkpoint_file", type=str, help="Path to the PyTorch checkpoint file.")
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.checkpoint_file, map_location='cpu')

        print_tensor_names(checkpoint)

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


