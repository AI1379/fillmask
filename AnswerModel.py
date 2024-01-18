import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, tensor, Tensor
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Any, AnyStr
from collections import namedtuple

modelCheckPoint = 'bert-large-cased'
preModel = AutoModelForMaskedLM.from_pretrained(modelCheckPoint)
preTokenizer = AutoTokenizer.from_pretrained(modelCheckPoint)

AnswerTuple = namedtuple("AnswerTuple", ['word', 'index'])

def TokenText(tokenizer, text):
    return tokenizer(text, return_tensors = "pt")

def PadSingleWord(
    model,
    tokenizer,
    text,
    options
) -> AnyStr:
    """
    根据上下文填充单个词语
    Args:
        model:          语言模型
        tokenizer:      编码器
        text:           上下文
        options:        给予的选项
    Returns:
        返回模型从给予的选项中概率最大
    """
    
    text = text.replace('_', '[MASK]')
    # 获得编码
    textToken = TokenText(tokenizer, text)
    textLogits = model(**textToken).logits

    # 获得掩码位置的概率
    maskTokenIndex = torch.where(textToken['input_ids'] == tokenizer.mask_token_id)[1]
    maskLogits = textLogits[0, maskTokenIndex, :][0]

    # 获得选项编码 并且去除头尾的标签
    optionsToken = TokenText(tokenizer, options)['input_ids'][0]
    optionsToken = optionsToken[1 : len(optionsToken) - 1]

    optionsLogits = tensor([maskLogits[index] for index in optionsToken])
    
    # 获得 答案下标
    resIndex = optionsLogits.argmax()

    return tokenizer.decode(optionsToken[resIndex]), resIndex

resIndex = PadSingleWord(preModel, preTokenizer, "This is _ model", "an fantastic Machillka a")

resIndex

def PadWholeText(
    model,
    tokenizer,
    text,
    options
) -> Any:
    """
    Args:
        model:
        tokenizer:
        text:           the text is filled with '[MASK]'
        options:        the options of text, shape: [QLength, OptionNum]
    Return:
        the text model completed and the tuple (word, index)
    """
    answer = []

    queryNum = len(options)

    # 进行全局编码
    textToken = TokenText(tokenizer, text)
    textLogits = model(**textToken).logits
    
    resTextToken = textToken['input_ids'][0]

    # 获得所有掩码位置
    maskTokenIndex = torch.where(textToken['input_ids'] == tokenizer.mask_token_id)[1]

    # 获得全局掩码理解
    maskLogits = textLogits[0, maskTokenIndex, :]

    # 如果 搜索到的 mask 和选项不匹配 则 断言error
    assert maskLogits.size(0) == queryNum, "mask shape is not euqal to options shape"

    # --- 根据全局理解进行填充
    for queryIdx in range(queryNum):
        # 当前查询的 掩码 概率
        logits = maskLogits[queryIdx]

        # 当前查询的 选项
        option = options[queryIdx]
        optionsToken = TokenText(tokenizer, option)['input_ids'][0]
        optionsToken = optionsToken[1 : len(optionsToken) - 1]
        
        optionsLogits = tensor([logits[index] for index in optionsToken])
    
        # 获得当前查询 答案下标
        resIndex = optionsLogits.argmax()
        word = tokenizer.decode(optionsToken[resIndex])

        answer.append(AnswerTuple(word, resIndex))

        resTextToken[maskTokenIndex[queryIdx]] = optionsToken[resIndex]

    # return answer, text
    return answer, tokenizer.decode(resTextToken[1 : len(resTextToken) - 1])

text = 'this is _, I _ your mom, Ahh _'
options =["it my mother", "love none test", "no name vision"]

answer, text = PadWholeText(preModel, preTokenizer, text, options)

print(*zip(*answer))
print(text)


