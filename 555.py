# -*- coding: utf-8 -*-
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import TextGenerationPipeline

model = GPT2LMHeadModel.from_pretrained("SkyWork/SkyText")
tokenizer = AutoTokenizer.from_pretrained("SkyWork/SkyText", trust_remote_code=True)
text_generator = TextGenerationPipeline(model, tokenizer, device=0)
input_str = "今天是个好天气"
max_new_tokens = 20
print(text_generator(input_str, max_new_tokens=max_new_tokens, do_sample=True))















#============qa


from transformers import pipeline
question_answerer = pipeline("question-answering", model='SkyWork/SkyText')

context = r"""
我叫沃尔夫冈，我住在柏林。
"""

result = question_answerer(question="我住在哪里",context=context)
print('==========')
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
print('==========')
print(result)







