import json
from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import argparse


class Prompt(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def check_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions_definition',
                        type=str,
                        dest="func_def",
                        default="data/input/functions_definition.json")

    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json")

    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calls.json")

    args = parser.parse_args()
    return {
        "func_defs": args.func_def,
        "input": args.input,
        "output": args.output
    }


def parser():
    parameters = check_flags()

    try:
        with open(parameters['func_defs'], "r") as f:
            func_def = json.load(f)
        with open(parameters['input'], 'r') as f:
            prompts_data = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{e}")
        exit(1)
    try:
        funcs = [FunctionDefinition.model_validate(item) for item in func_def]
    except ValidationError:
        print("[ERROR] Invalid funcion definition:")
    try:
        prompts = [Prompt.model_validate(item) for item in prompts_data]
    except ValidationError:
        print("[ERROR] Invalid prompts input!")


parser()
