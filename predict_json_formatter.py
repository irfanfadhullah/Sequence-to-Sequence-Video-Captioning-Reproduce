import json, os
import pandas as pd

with open('prediction_dict_fix.json', '+rb') as f: #change to your file path, this file is created from evaluation.py
    a = json.load(f)

a = pd.DataFrame(a.items(), columns=['name', 'caption'])
list_index = list(range(1300,1970))

a['index'] = list_index

pred_key = list(a['index'].values)
pred_val = list(a['caption'].values)
pred_val_fix = [[data] for data in pred_val]

pred_key_fix = [f"{data}" for data in pred_key]
pred_dict = dict(zip(pred_key_fix, pred_val_fix))

dict_fix = {
    "email":"xxxxx@gxxx.com",
    "predictions": pred_dict
}
with open('predict_result.json', '+w') as f : #change to your directory
    json.dump(dict_fix, f)