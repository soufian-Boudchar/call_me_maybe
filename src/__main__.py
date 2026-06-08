from llm_sdk import Small_LLM_Model
from parsing import parser


prompt = "What is the sum of 10 and 20"
llm = Small_LLM_Model()

input_ids = llm.encode(prompt).squeeze().tolist()
logits = llm.get_logits_from_input_ids(input_ids)


print(logits)
