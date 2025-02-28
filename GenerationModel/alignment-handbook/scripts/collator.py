# from transformers import DataCollatorForLanguageModeling

# import torch
# class DataCollatorForCompletionOnlyLM_SELF(DataCollatorForLanguageModeling):
#     """
#     Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
#     when they do not come from the assistant. This ensure that the loss is only
#     calculated on the completion made by the assistant.

#     Args:
#         response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
#             '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
#             differently if it does not have proper context.
#         instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
#             '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
#         mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
#             `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
#              for flexibility and backwards-compatibility.
#         ignore_index (`int`, *optional*, defaults to `-100`):
#             The index to use to ignore the initial tokens with
#     """

#     def __init__(
#         self,
#         *args,
#         mlm: bool = False,
#         ignore_index: int = -100,
#         **kwargs,
#     ):
#         super().__init__(*args, mlm=mlm, **kwargs)
#         self.special_token = {
#                     "begin":torch.tensor([128000, 128000, 128006, 9125, 128007]), #<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>
#                     "user":torch.tensor([128006, 882, 128007]),#<|start_header_id|>assistant<|end_header_id|>
#                     "assistant":torch.tensor([128006, 78191, 128007,271]),#<|start_header_id|>assistant<|end_header_id|>\n\n
#                     "end":torch.tensor([128009]),#<|eot_id|>
#                     "sub_retrieval":torch.tensor([12289, 7379, 838, 1363]) #<Retrieval>\n\n的一部分
#                 }
#         self.ignore_index = ignore_index
    

#     def torch_call(self, examples):
#         """
#         examples:[{'input_ids':,'attention_mask':},{'input_ids':,'attention_mask':}]
#         """
#         #batch返回的是：{"input_ids":[[]],"attention_mask":[[]],"labels":[[]]}
#         batch = super().torch_call(examples)
#         self.eot = torch.tensor([-100])
#         for i in range(len(batch['labels'])):

#             #先找到eot的位置
#             eot_site_index = torch.where(batch["labels"][i] == self.eot)[0]

#             #因为pad = eot,所以还要更改，将后面因为扩充导致的-100删除
#             max_length = len(eot_site_index)
#             index = 0
#             while index + 1 < max_length:
#                 if eot_site_index[index] + 1 == eot_site_index[index + 1]:
#                     eot_site_index = eot_site_index[:index+1]
#                     break
#                 else:
#                     index += 1
       
#             #判断有没有system_prompt
#             if torch.equal(batch['labels'][i][:len(self.special_token['begin'])],self.special_token['begin']):
#                 ignore_index = 1
#             else:
#                 ignore_index = 0
#             #第一句对话的标记为-100
#             batch["labels"][i][:eot_site_index[ignore_index] + len(self.special_token['assistant']) + 1] = self.ignore_index
           
#             #开始在回答的信息中找<Retrieval>\n\n和</Retrieval>\n\n
#             search_begin_index = eot_site_index[ignore_index] + len(self.special_token['assistant']) + 1
#             search_end_index = eot_site_index[ignore_index + 1]
#             index = search_begin_index
#             retrieval_list_end_index = []
           
#             while index + len(self.special_token['sub_retrieval']) <= search_end_index + 1:
#                 if torch.equal(batch['labels'][i][index:index + len(self.special_token['sub_retrieval'])],
#                                self.special_token['sub_retrieval']):
#                     retrieval_list_end_index.append(index + len(self.special_token['sub_retrieval']) - 1)
#                 index += 1

#             while(ignore_index + 3 < len(eot_site_index)):
#                 batch['labels'][eot_site_index[ignore_index + 1] + 1 : eot_site_index[ignore_index + 2] + len(self.special_token['assistant']) + 1] = self.ignore_index
#                 ignore_index += 2

#                 search_begin_index = eot_site_index[ignore_index] + len(self.special_token['assistant']) + 1
#                 search_end_index = eot_site_index[ignore_index + 1] 

#                 while index + len(self.special_token['sub_retrieval']) <= search_end_index + 1:
#                     if torch.equal(batch['labels'][i][index:index + len(self.special_token['sub_retrieval'])],
#                                     self.special_token['sub_retrieval']):
#                         retrieval_list_end_index.append(index + len(self.special_token['sub_retrieval']) - 1)
#                     index += 1

#             for index in range(0,len(retrieval_list_end_index),2):
#                 batch['labels'][i][retrieval_list_end_index[index] + 1 :retrieval_list_end_index[index + 1] + 1] = self.ignore_index
#         return batch

import torch,time
from transformers import DataCollatorForLanguageModeling

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
        self.special_token = {
                    "begin":torch.tensor([128000, 128000, 128006, 9125, 128007]), #\n<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    "user":torch.tensor([128006, 882, 128007]),#<|start_header_id|>assistant<|end_header_id|>
                    "assistant":torch.tensor([128006, 78191, 128007,271]),#<|start_header_id|>assistant<|end_header_id|>\n\n
                    "end":torch.tensor([128009]),#<|eot_id|>
                    "sub_retrieval":torch.tensor([12289, 7379, 838, 1363]) #<Retrieval>\n\n的一部分,
                }
        self.ignore_index = ignore_index
    

    def torch_call(self, examples):
        """
        examples:[{'input_ids':,'attention_mask':},{'input_ids':,'attention_mask':}]
        """
        #batch返回的是：{"input_ids":[[]],"attention_mask":[[]],"labels":[[]]}
        batch = super().torch_call(examples)
        # print(batch)
        # print(self.tokenizer.decode(batch['input_ids'][0]))
        self.eot = torch.tensor([-100])
        for i in range(len(batch['labels'])):

            #先找到eot的位置
            eot_site_index = torch.where(batch["labels"][i] == self.eot)[0]

            #因为pad = eot,所以还要更改，将后面因为扩充导致的-100删除
            max_length = len(eot_site_index)
            index = 0
            while index + 1 < max_length:
                if eot_site_index[index] + 1 == eot_site_index[index + 1]:
                    eot_site_index = eot_site_index[:index+1]
                    break
                else:
                    index += 1
            
            # print(eot_site_index)

            # print("batch_labels: ")
            # print(batch['labels'][i])

            #判断有没有system_prompt
            if torch.equal(batch['labels'][i][:len(self.special_token['begin'])],self.special_token['begin']):
                ignore_index = 1
            else:
                ignore_index = 0

            batch["labels"][i][:eot_site_index[ignore_index] + len(self.special_token['assistant']) + 1] = self.ignore_index
            #将user说的话进行-100标记
            for eot_index in range(ignore_index + 2,len(eot_site_index),2):
                batch["labels"][i][eot_site_index[eot_index - 1] + 1:eot_site_index[eot_index] + len(self.special_token["assistant"]) + 1] = self.ignore_index
            
           
            #开始在回答的信息中找<Retrieval>\n\n和</Retrieval>\n\n
            search_begin_index_list = []
            search_end_index_list = []

            for assistant_index in range(ignore_index,len(eot_site_index),2):
                search_begin_index_list.append(eot_site_index[assistant_index] + len(self.special_token["sub_retrieval"]) + 1)
                try:
                    search_end_index_list.append(eot_site_index[assistant_index + 1])
                except:
                    print("ingnore_index:  ",ignore_index)
                    print("len_of_eot_index: ",len(eot_site_index))
                    print("句子是： ",self.tokenizer.decode(batch["input_ids"][i]))
                    raise ValueError("数据有问题")

            retrieval_res_list = []
            for retrieval_search_index in range(len(search_begin_index_list)):
                index = search_begin_index_list[retrieval_search_index]
                search_end_index = search_end_index_list[retrieval_search_index]
               
                while index + len(self.special_token['sub_retrieval']) - 1 <= search_end_index:
                   
                    if torch.equal(batch['labels'][i][index:index + len(self.special_token['sub_retrieval'])],
                                    self.special_token['sub_retrieval']):
                        retrieval_res_list.append(index + len(self.special_token['sub_retrieval']) - 1)
                    index += 1

            if len(retrieval_res_list) % 2 != 0:
                print("Retrieval 个数不匹配，下面是数据")
                print(self.tokenizer.decode(batch['input_ids'][i]))
                raise ValueError("查找到的retrieval个数不匹配")
            for retrieval_index in range(0,len(retrieval_res_list),2):
                batch['labels'][i][retrieval_res_list[retrieval_index] + 1:retrieval_res_list[retrieval_index + 1]+1] = self.ignore_index
            # 对句子中的eot进行更改
            for eot_index in range(ignore_index + 1, len(eot_site_index),2):
                batch['labels'][i][eot_site_index[eot_index]] = self.tokenizer.eos_token_id

        self_sl_max_length = 7168
        batch['input_ids'] = batch['input_ids'][:,:self_sl_max_length]
        batch['attention_mask'] = batch['attention_mask'][:,:self_sl_max_length]
        batch['labels'] = batch['labels'][:,:self_sl_max_length]
            # 校验
            # for index in range(len(batch['labels'][i])):
            #     if (batch['labels'][i][index] == -100):
            #         batch['labels'][i][index] = 1
            # print("________begin_origin________")
            # print(self.tokenizer.decode(batch['input_ids'][i]))
            # print("_____________labels_begin________________")
            # print(batch['labels'][i])
            # print("_____________labels_end________________")
            # print(self.tokenizer.decode(batch['labels'][i]))
            # print("________end_new______________")

        return batch
    
   