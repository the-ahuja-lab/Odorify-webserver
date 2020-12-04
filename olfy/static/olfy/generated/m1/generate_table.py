import pandas as pd
import sys
# import sys
# sys.argv[1]
# sys.argv2[2]
path = sys.argv[1]
test_pred = pd.read_csv(f'{path}/results.csv')
input_smile = pd.read_csv(f'{path}/input.csv')
test_pred["prob"] = test_pred["property"].apply(lambda x: x if x> 0.5 else (1-x)) 
input_smile["prob"]=test_pred["prob"]
input_smile["pred_odor"]=test_pred["property"].round(0)
for i in range (len(input_smile)):
	input_smile["prob"][i]=str(input_smile["prob"][i])[:5]
# test_true=test_true.drop(columns=['odor'])
predicted_output = input_smile
predicted_output.to_csv(f"{path}/predicted_output.csv")

