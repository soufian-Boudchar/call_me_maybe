import json
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List
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


def check_flags() -> Dict[str, str]:
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
        "funcs_def": args.func_def,
        "input": args.input,
        "output": args.output
    }

def prompts_parser(prompts_path: Dict[str, str]) -> List[Prompt]:
    parameters = prompts_path
    try:
        with open(parameters['input'], 'r') as f:
            prompts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{e}")
        exit(1)

    try:
        prompts = [Prompt.model_validate(item) for item in prompts]
    except ValidationError:
        print("[ERROR] Invalid prompts input!")
        exit(1)
        
    return prompts
    

def funcs_parser(funcs_path: Dict[str, str]) -> List[FunctionDefinition]:
    parameters = funcs_path

    try:
        with open(parameters['funcs_def'], "r") as f:
            func_def = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{e}")
        exit(1)

    try:
        funcs = [FunctionDefinition.model_validate(item) for item in func_def]
    except ValidationError:
        print("[ERROR] Invalid funcion definition:")
        exit(1)
        
    return funcs
