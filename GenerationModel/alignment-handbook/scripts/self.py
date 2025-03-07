import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from trl import DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling,StoppingCriteria,AutoTokenizer
import numpy as np
import sys
import json
import torch
from collections.abc import Mapping
import copy

 #1、有几个特殊的tokens，1个是<|system|>\n <|user|>\n <|assistant|>\n </s>\n <Retrieval> <Retrieval/>\n
 #2、从<|system|>\n 到 第一个<|assistant|>\n的部分要标记为-100，每一个例子中有且仅有一个
 #3、如果是多轮对话，其他的从<|user|>\n到<|assistant|>\n部分也要进行-100标记
 #4、<|assistant|>\n的标签为-100
 #5、<Retrieval>需要训练，（<Retrieval>，<Retrieval/>]之间的标记需要标记为-100



class DataCollatorForCompletionOnlyLM_SELF(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        
        self.template_dict = {'begin_of_text':'<|begin_of_text|>',      #128000
                              'start_header_id':'<|start_header_id|>',  #128006
                              'end_header_id':'<|end_header_id|>',      #128007
                              'eot_id':'<|eot_id|>',                #128009
                              '':'</Retrieval>\n'  
                              }
        
        self.ignore_index = ignore_index

        for key,value in self.template_dict.items():
            setattr(self,key+'_ids',self.tokenizer.encode(self.template_dict[key],add_special_tokens=False))


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        '''
        输入的examples是[{'input_ids':,'attention_mask':},{'input_ids':,'attention_mask':},{'input_ids':,'attention_mask':}]

        从[0:第一个assitant\n]是标记为-100的，所以这里的system 第一个user，加第一个assitant已经被标记被-100了
        后面的所有的[user,assiatant]都要标记为-100
        提示词里面有一个对[retrieval，retrieval/]要标记为-100
        如果发生了检索：
            其余的成对的(retrieval,Retrieval/]表示为-100

        不用</s>\n的原因是，其实他是eos_token,当self.mlm = False的时候，labels的pad_token都会被标记为-100，警惕pad_token = eos_token
        '''
        batch = super().torch_call(examples) 
        max_length = 8192
        #执行上面的代码之后会添加一个labels的标签。
        # print('batch: ',batch)
        for index in range(len(batch['input_ids'])):
            #如果是eos结尾或者填充结尾，说明这个至少是完整的一段话
            begin_index = torch.where(batch["input_ids"][index] == 128006)[0]
            eos_index = torch.where(batch['input_ids'][index] == 128009)[0]
            if batch['labels'][index][-1] == -100 or batch['labels'][index][-1] == 128009:
                #这是完整的一段话
                #有system prompt的模板
                if torch.equal(batch['input_ids'][index][:5], torch.tensor([128000, 128006, 9125, 128007, 271])):
                    batch['labels'][index][:eos_index[0]+1] = -100    #将<begin_of_text>到第一个<eos_id>的位置标记为-100

                    #开始标记user的对话
                    for begin in range(1,len(begin_index),2):
                        left = begin_index[begin]
                        right = min(eos_index[begin]+4+1,max_length)
                        batch['labels'][index][left:right] = -100
                    
                #没有system prompt
                else:
                    batch['labels'][index][0] = -100
                    for begin in range(0,len(begin_index),2):
                        left = begin_index[begin]
                        right= min(eos_index[begin] + 4 + 1, max_length)
                        batch['labels'][index][left:right] = -100

            else:
                
                #这是被截断的
                if torch.equal(batch['input_ids'][index][:5], torch.tensor([128000, 128006, 9125, 128007, 271])):
                    if len(eos_index) == 0:
                        batch['labels'][index][:] = -100
                    else:
                        batch['labels'][index][:eos_index[0] + 1] = -100

                        for i in range(1,len(eos_index),2):
                            left = begin_index[i]
                            right = min(eos_index[i] + 4 + 1, max_length)
                            batch['labels'][index][left:right] = -100
                        
                        if len(begin_index) % 2 == 0:
                            left = begin_index[-1]
                            right = max_length
                            batch['labels'][index][left:right] = -100
                
                else:
                    batch['labels'][index][0] = -100

                    if len(eos_index) == 0:
                        batch['labels'][index][:] = -100
                    else:
                        for i in range(0,len(eos_index),2):
                            left = begin_index[i]
                            right = min(eos_index[i] + 4 + 1, max_length)
                            batch['labels'][index][left:right] = -100

                        if len(begin_index) % 2 != 0 :
                            left = begin_index[-1]
                            right = max_length
                            batch['labels'][index][left:right] = -100
        return batch


#如果你使用一下的方案，这个代码只支持一个batch

class KeyWordOne_StoppingCriteri(StoppingCriteria):
    def __init__(self,keyword,tokenizer,device):
        self.keyword = tokenizer.encode(keyword,add_special_tokens=False,return_tensors='pt').squeeze().to(device)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) < len(self.keyword):
            return False
        print(input_ids[0][len(input_ids[0]) - len(self.keyword):])
        if input_ids[0][len(input_ids[0]) - len(self.keyword):].equal(self.keyword):
            return True
        return False



###test###

if __name__ == "__main__":
    model_path = "meta-llama/Meta-Llama-3-8B"
    path = '/home/lsh/code/alignment-handbook/dataset_other/20tasks 2/1-2F.json'

    tokenizer = AutoTokenizer.from_pretrained(model_path,token = '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')
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(['<Retrieval>','<Retrieval/>\n'])

    data = json.load(open(path,'r'))
    data_example1 = data[0]
    data_example2 = data[1]
    string1 = tokenizer.apply_chat_template(data_example1,tokenize=False)
    string2 = tokenizer.apply_chat_template(data_example2,tokenize=False)
    examples = [tokenizer(string1),tokenizer(string2)]
    collator_fn = DataCollatorForCompletionOnlyLM_SELF(tokenizer = tokenizer)
    batch = collator_fn(examples)
    print(batch)

