import datasets
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--number", type=int, default=1)
args = parser.parse_args()

test_file = ["", "data", "data2", "data3"]
parquet_file = f'{test_file[args.number]}/final_data.{args.number}.parquet' #'./test2/final_data.2.parquet' #'./test3/final_data.3.parquet'
table = pq.read_table(parquet_file)
df = table.to_pandas()
df.to_json('final_data.json', orient='records', lines=True, force_ascii=False)
final_data = {}
data_idx1 = [("mmlu", 1129), ("truthfulqa", 9116), ("truthfulqa", 15705), ("bbq", 41312), ("cnn", 42212), ("gsm8k", 49685), ("chat", 51495)]
data_idx2 = [("mmlu", 1129), ("truthfulqa", 9116), ("truthfulqa", 15705), ("bbq", 73197), ("cnn", 74097), ("gsm8k", 81570), ("chat", 83380)]
data_idx3 = [("mmlu", 1129), ("truthfulqa", 9116), ("truthfulqa", 15705), ("bbq", 73197), ("cnn", 74097), ("gsm8k", 81570), ("chat", 83380)]
data_idx = data_idx1
with open("final_data.json", "r") as f:
    num = -1
    now = 0
    final_data[data_idx[now][0]] = [[]]
    for i in f:
        num += 1
        if num == data_idx[now][1]:
            now += 1
            if data_idx[now][0] not in final_data:
                final_data[data_idx[now][0]] = []
            final_data[data_idx[now][0]].append([])
        final_data[data_idx[now][0]][-1].append(i)

for k in final_data.keys():
    tot = 0
    for item in final_data[k]:
        print(k, len(item))
        output_file = f'./{test_file[args.number]}/{k}_{len(item)}.json'
        fr = open(output_file, "w")
        for i in item:
            fr.write(i)
        fr.close()