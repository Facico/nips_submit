from typing import Dict, List
import pickle

word_list = pickle.load(open("utils/post_process.bin", "rb"))
GENDER_TO_WORD_LISTS = word_list["GENDER_TO_WORD_LISTS"]
RACE_TO_NAME_LISTS = word_list["RACE_TO_NAME_LISTS"]
ADJECTIVE_LIST = word_list["ADJECTIVE_LIST"]
PROFESSION_LIST = word_list["PROFESSION_LIST"]