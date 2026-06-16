from parsing.parser import prompts_parser, funcs_parser, check_flags
from llm_sdk import Small_LLM_Model




def main() -> None:
    args = check_flags()
    prompts = prompts_parser(args)
    funcs = funcs_parser(args)
    
    print(prompts[0])
    # for i in prompts:
    #     print()
    
    
    
if __name__ == "__main__":
    main()
