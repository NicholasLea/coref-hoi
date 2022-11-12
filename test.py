import re

import pyhocon
from transformers import BertModel

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

# config_name = 'train_spanbert_large_ml0_d2'
# config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]
# print("config['bert_pretrained_name_or_path']", config['bert_pretrained_name_or_path'])
#
# # bert = BertModel.from_pretrained('./data_dir/spanbert_base')
# bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])


# 可运行
# from transformers import BertTokenizer,BertModel,BertConfig
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = BertConfig.from_pretrained('bert-base-uncased')
# config.update({'output_hidden_states':True})
# model = BertModel.from_pretrained("bert-base-uncased",config=config)
# print(model)


import subprocess
cmd = ["conll-2012\\scorer\\v8.01\\scorer.pl", 'muc', './data_dir/test.english.v4_gold_conll', "C://Users//59876//AppData//Local//Temp//tmpgtgz5qic", "none"]
# cmd = ["conll-2012\\scorer\\v8.01\\scorer.pl"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
print('process:', process)
stdout, stderr = process.communicate(b'\n')
print('stderr:', stderr)

process.wait()
print('stdout:', stdout)

stdout = stdout.decode("utf-8")

print('stdout:', stdout)
coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
print('coref_results_match:', coref_results_match)
recall = float(coref_results_match.group(1))
precision = float(coref_results_match.group(2))
f1 = float(coref_results_match.group(3))