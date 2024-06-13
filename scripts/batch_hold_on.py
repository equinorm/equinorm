import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForMaskedLM

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained("/apdcephfs/share_1594716/bingzhewu/pretrained/bert-base-chinese", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("/apdcephfs/share_1594716/bingzhewu/pretrained/bert-base-chinese", trust_remote_code=True).to(device)



raw_inputs = [
    'Who are you?',
'Who are you?'
]

while True:
    input_ids = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt").to(device)

    gen_kwargs = {'max_length': 100, 'do_sample': True, 'top_p': 0.9, 'top_k': 0, 'temperature': 0.95, 'num_return_sequences': 1}
    outputs = model.generate(**input_ids, **gen_kwargs)
    result = tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
