import os
import pandas as pd
import numpy as np

results_path = '../outputs/results_for_ansambling/'
results_path = os.path.abspath(results_path)

merged_results = []

for file_name in os.listdir(results_path):
    full_path = os.path.join(results_path, file_name)
    with open(full_path, 'r') as f:
        for i, line in enumerate(f):
            class_name, prob = line.strip().split(',')
            prob = float(prob)
            if len(merged_results) > i:
                merged_results[i].append((class_name, prob))
            else:
                merged_results.append([(class_name, prob)])
                
final_results = []
# for res in merged_results:
#     res1, res2 = res
#     if res1[0] != res2[0]:
#         final_results.append('Other')
#     else:
#         final_results.append(res1[0])

cnt_different = 0
confidence_threshold = 0.85
# confidence_threshold = 0.6
for i, res in enumerate(merged_results):
    class_votes = []
    confidence = []
    for result in res:
        class_votes.append(result[0])
        confidence.append(result[1])

    confidence = np.array(confidence)
    class_votes = np.array(class_votes)
    class_name, votes = np.unique(class_votes, return_counts = True)
    max_ind = np.argmax(votes)

    max_vote_indices = np.where(class_votes==class_name[max_ind])
    other_vote_indices = np.where(class_votes!=class_name[max_ind])
    conf_subset = confidence[max_vote_indices]
    conf_oposite_subset = confidence[other_vote_indices]
    # if votes[max_ind] == len(class_votes) and len(conf_subset[conf_subset > confidence_threshold]) >= np.ceil(len(class_votes)/2): 
    if votes[max_ind] >= np.ceil(len(class_votes)/2) and len(conf_subset[conf_subset > confidence_threshold]) >= np.ceil(len(class_votes)/2): 
    # if votes[max_ind] >= np.ceil(len(class_votes)/2) and (len(conf_oposite_subset[conf_oposite_subset > confidence_threshold]) < len(conf_oposite_subset) or len(conf_oposite_subset)==0): 
        # print(i,res, votes[max_ind], len(conf_subset[conf_subset > confidence_threshold]) >= np.ceil(len(class_votes)/2))
        # print(i,"TOCNO ", class_name[max_ind], res)
        final_results.append(class_name[max_ind])
    else:
        # print(i,"NETOCNO ", "Other", res)
        cnt_different += 1
        final_results.append('Other')

sub = pd.DataFrame()
sub['Results'] = final_results
sub.to_csv('Mozgalo.csv', index=False, header=False)
print(len(final_results), "Others = %d" % cnt_different)
    