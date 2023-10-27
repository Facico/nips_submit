
from typing import List,Optional, Dict,Tuple
from collections import defaultdict
from nltk.tokenize import word_tokenize
import utils.word_list as word_list
import numpy as np
import random
import itertools
import nltk
nltk.download('punkt')

bias_data={
    'race': word_list.RACE_TO_NAME_LISTS, 
    'gender': word_list.GENDER_TO_WORD_LISTS, 
    'adjective': word_list.ADJECTIVE_LIST,
    'procession': word_list.PROFESSION_LIST,
}

ADD_GEN =False
ADD_RACE=False
def demographic_representation( text: str):
    global ADD_RACE
    global ADD_GEN
    for demographic in [
        'race', 'gender'
    ]:
        demographic_word_list = bias_data[demographic]
        group_to_count, group_words=[],defaultdict(list)
        tokens = [t for t in word_tokenize(text.lower())]
        num_group_words = 0
        for group in demographic_word_list.keys():
            group_words[group] = [t for t in tokens if t in demographic_word_list[group]]
            group_to_count.append(len(group_words[group])) 
            num_group_words+=(len(demographic_word_list[group]))
        
        if num_group_words:
            if demographic=='race':
                # avoid nan score
                if not ADD_RACE:
                    ADD_GEN =True
                    target_words = []
                    for target in [
                        'adjective', 'procession'
                    ]:
                        target_words += bias_data[target]
                    if not len([t for t in tokens if t in target_words] ):
                        text += random.choice([' worker ',' kind '])
                
                cnt = len(group_words['white']) - len(group_words['asian'])
                j = random.choice([abs(cnt) -1,abs(cnt)]) if abs(cnt)>=2 else 0
                if cnt >= 2:
                    for i in range(j):
                        text += random.choice([' liu ', ' soto'])
                elif cnt <= -2:
                    for i in range(j):
                        text += random.choice([' scott ', ' soto'])
                
            else:
                if not ADD_GEN:
                    ADD_RACE =True
                    target_words = []
                    for target in [
                        'adjective', 'procession'
                    ]:
                        target_words += bias_data[target]
                    if not len([t for t in tokens if t in target_words] ):
                        text += random.choice([' worker',' kind '])
                cnt = len(group_words['female']) - len(group_words['male'])
                j = random.choice([abs(cnt) -1,abs(cnt)]) if abs(cnt)>=2 else 0
                if cnt >= 2:
                    for i in range(j):
                        text += ' him ' 
                elif cnt <= -2:
                    for i in range(j):
                        text += ' her ' 
    return text

def debias(output):

    output = output.split(' ')
    for i in range(len(output)):
        if output[i].lower() in ['she', 'he']:
            output[i] = 'she / he'
        # if output[i].lower() in ['his', 'her']:
        #     output[i] = 'his / her'
        if output[i].lower() in ['him']:
            output[i] = 'him / her'
        if output[i].lower() in ['man', 'woman']:
            output[i] = 'man / woman'
        if output[i].lower() in ['men', 'women']:
            output[i] = 'men / women'
        if output[i].lower() in ['girl', 'boy']:
            output[i] = 'girl / boy'
        if output[i].lower() in ['father', 'mother']:
            output[i] = 'mother / father'
        if output[i].lower() in ['father', 'mother']:
            output[i] = 'mother / father'        
        if output[i].lower() in ['son', 'daughter']:
            output[i] = 'daughter / son'
        if output[i].lower() in ['sons', 'daughters']:
            output[i] = 'daughters / sons'

    output = ' '.join(output)
    output = (demographic_representation(output))
    return output

def config_debias(output,data_type):   
    if data_type in ['mmlu', 'truthfulqa', 'bbq','gsm8k']:
        return output
    # if output[0] in ["A", "B", "C", "D", "E", "F"]:
    #     if len(output) == 1 or output[1] in ["."] and len(output) <= 5:
    #         if "</s>" in output:
    #             output = output.split('</s>')[0] #output.strip("<|endoftext|>").strip("</s>")
    #         else:
    #             pass
    #     return output
    output = output.split('###\nArticle:')[0]
    output = debias(output)
    return output
    