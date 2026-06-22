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
    

def main():
    llm = Small_LLM_Model(device="cpu")
    flags = check_flags()
    prompts = prompts_parser(flags)
    funcs = funcs_parser(flags)

    forced_prompt = prompts[1].prompt + '\n{"name": "'
    vocab_path = llm.get_path_to_vocab_file()
    vocab = get_llm_vocab(vocab_path)
    
    current_input_ids = llm.encode(forced_prompt).squeeze().tolist()
    quote_token_id = llm.encode('"').squeeze().tolist()

    print(forced_prompt, end="")
    while True:
        logits = llm.get_logits_from_input_ids(current_input_ids)
        best_token_id = logits.index(max(logits))
        decode_best_token = llm.decode(best_token_id)
        print(decode_best_token, end="")
        if '"' in decode_best_token:
            break
        current_input_ids.append(best_token_id)

if __name__ == "__main__":
    main()