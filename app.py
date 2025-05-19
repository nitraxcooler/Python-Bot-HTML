import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("justdanitraxxd/pythonbot-model")
tokenizer = AutoTokenizer.from_pretrained("justdanitraxxd/pythonbot-model")

def chat(input_text):
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="PythonBot")
iface.launch()
