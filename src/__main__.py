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

def main():

    llm = Small_LLM_Model(device="cpu")
    flags = check_flags()
    prompts = prompts_parser(flags)
    funcs = funcs_parser(flags)


    valid_names = get_funcs_name(funcs)

    system_context = get_system_context(funcs)
    user_question = prompts[-1].prompt
    forced_prompt = system_context + "\nQuestion:" + user_question + '\n{"name": "'
    vocab_path = llm.get_path_to_vocab_file()
    vocab = get_llm_vocab(vocab_path)
    
    current_input_ids = llm.encode(forced_prompt).squeeze().tolist()
    

    generated_text = "" 

    print(forced_prompt, end="")
    
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
        
        print(decode_best_token, end="", flush=True)
        
        if '"' in decode_best_token:
            break
            
        generated_text += decode_best_token
        current_input_ids.append(best_token_id)

if __name__ == "__main__":
    main()