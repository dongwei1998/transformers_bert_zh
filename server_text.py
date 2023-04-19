# coding=utf-8
# =============================================
# @Time      : 2023-04-16 10:42
# @Author    : DongWei1998
# @FileName  : server_text.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
from transformers import pipeline
# Replace this with your own checkpoint
model_checkpoint = "bert-finetuned-squad_zh"
question_answerer = pipeline("question-answering", model=model_checkpoint,framework='tf')
# question_answerer = pipeline("fill-mask", model=model_checkpoint, tokenizer=model_checkpoint)
context = """
 1、十月革命胜利,世界上出现了第一个社会主义国家.一个崭新的社会主义报刊体系在苏俄确立形成.<e>2、二战结束后,又有欧、亚、拉美一系列国家脱离了
资本主义体系,走社会主义道路,社会主义报业得到很大发展.<e>3、苏东”剧变后,这些国家的报业结构和性质发生了重大变化.<e>十六、苏联时期报刊体制的主要特征是 怎样的?<e>1、苏联的报刊,都属于国家所有,是党和国家机构的重要组成部分；其基本职能是集体的宣传员、集体的鼓动员和集体的组织者.<e>2、苏联的各级报刊绝对服从于各级党委的领导.<e>3、苏联报纸信息来源单一,言论高度集中.<e>4、苏联报刊在建设时期是社会主义建设的工具.<e>十七、发展中国家报业又何共同特点?<e>1、早期报刊、尤其是报业发端较早的国家的早期报刊,大多是殖民者创办的；<e>2、随着反殖民主义反封建斗争的开展,这些国家的民族报刊逐步发展起来,并推动了反殖民主 义反封建斗争的进程；<e>3、民族解放运动胜利后,大多数报业获得了前所未有的发展,但也有的国家报业重新陷入本国独裁者的控制之下.<e>十八、新闻通讯社是在怎样的背景下诞生的?它的功能与作用如何?
"""
question = "世界上最早的报纸诞生于"

result = question_answerer(question=question, context=context)
if result['score'] < 0.5:
    print(f"Sorry, I don't know how to represent {question}.")
else:
    print(f"{question} is {result['answer']}.")
