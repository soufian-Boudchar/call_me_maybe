from parsing.parser import prompts_parser, funcs_parser, check_flags
from llm_sdk import Small_LLM_Model
import json




def get_llm_vocab(path: str):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            vocab = json.load(f)
    except Exception as e:
        print(f"[ERROR] {e}")
    return vocab



def get_funcs_name(data):
    return [f.name for f in data]



def get_system_context(funcs):
    system_context = "You are a helpful assistant. You have access to the following functions:\n"
    for f in funcs:
        system_context += f"- {f.name}: {f.description}\n"
    return system_context

def main(user_question, llm):
    
    # get flags from terminal
    flags = check_flags()
    
    # parse func defs and prompts from input files
    prompts = prompts_parser(flags)
    funcs = funcs_parser(flags)

    # get list of valid functions names ['fn_add_numbers', ...]
    valid_names = get_funcs_name(funcs)

    vocab_path = llm.get_path_to_vocab_file()
    vocab = get_llm_vocab(vocab_path)

    system_context = get_system_context(funcs)

    # merging user quetion with llm message and add {"name": to make llm complete the name of function
    forced_prompt = system_context + "\nQuestion:" + user_question + '\n{"name": "'
    
    # extract tokenst to tensor 
    current_input_ids = llm.encode(forced_prompt).squeeze().tolist()
    
    generated_text = ""

    while True:
        logits = llm.get_logits_from_input_ids(current_input_ids)
        original_logits = list(logits) 

        for i in range(len(logits)):
            logits[i] = float('-inf')
            
        for token_string, token_id in vocab.items():
            test_string = generated_text + token_string
            is_valid = False
            
            for name in valid_names:
                if name.startswith(test_string):
                    is_valid = True
                    break
            
            if generated_text in valid_names and '"' in token_string:
                is_valid = True
                
            if is_valid:
                if token_id < len(logits):
                    logits[token_id] = original_logits[token_id]

        best_token_id = logits.index(max(logits))
        decode_best_token = llm.decode([best_token_id])
        
        # 7yydna print l'sghir hna
        
        if '"' in decode_best_token:
            break
            
        generated_text += decode_best_token
        current_input_ids.append(best_token_id)
    
    selected_func = next((f for f in funcs if f.name == generated_text), None)
    
    if not selected_func:
        print("[ERROR] Function not found!")
        exit(1)

    json_result = f'{{"name": "{generated_text}", "parameters": {{'
    # 7yydna print l'sghir hna
    
    current_input_ids = llm.encode(forced_prompt + f'{generated_text}", "parameters": {{').squeeze().tolist()

    params = list(selected_func.parameters.keys())
    
    for i, param_name in enumerate(params):
        param_type = selected_func.parameters[param_name]["type"]
        
        comma = ", " if i > 0 else ""
        param_prefix = f'{comma}"{param_name}": '
        json_result += param_prefix
        
        if param_type == "string":
            json_result += '"'
            param_prefix += '"'
            
        current_input_ids += llm.encode(param_prefix).squeeze().tolist()

        value_generated = ""
        while True:
            logits = llm.get_logits_from_input_ids(current_input_ids)
            original_logits = list(logits)
            
            for j in range(len(logits)):
                logits[j] = float('-inf')
                
            for token_string, token_id in vocab.items():
                is_valid = False
                
                if param_type == "number":
                    if all(char.isdigit() or char == '.' or char == ' ' for char in token_string):
                        is_valid = True
                    if token_string.strip() in [',', '}', '']: 
                        is_valid = True
                        
                elif param_type == "string":
                    is_valid = True 
                    if "<" in token_string or ">" in token_string: 
                        is_valid = False

                if is_valid and token_id < len(logits):
                    logits[token_id] = original_logits[token_id]

            best_token_id = logits.index(max(logits))
            token_str = llm.decode([best_token_id])
            
            if param_type == "number" and (',' in token_str or '}' in token_str or '\n' in token_str):
                break 
            if param_type == "string" and '"' in token_str:
                json_result += '"' 
                break 
                
            # 7yydna print l'sghir hna
            value_generated += token_str
            json_result += token_str
            current_input_ids.append(best_token_id)
            
    json_result += '}}'

    print(json_result)


if __name__ == "__main__":
    
    llm = Small_LLM_Model(device="cpu")
    flags = check_flags()
    

    prompts = prompts_parser(flags)
    funcs = funcs_parser(flags)
    for p in prompts:
        main(p.prompt, llm)