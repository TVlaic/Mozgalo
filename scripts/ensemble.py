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
confidence_threshold = 0.95 #bilo 0.95
cnt_same = 0
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

    # required_number_of_votes = np.floor(len(class_votes)/2) #Za isprobat sutra s folderom spremnim
    required_number_of_votes = np.ceil(len(class_votes)/2)
    # if votes[max_ind] >= required_number_of_votes and len(conf_subset[conf_subset > confidence_threshold]) >= required_number_of_votes: 
    if (votes[max_ind] >= required_number_of_votes and len(conf_subset[conf_subset > confidence_threshold]) >= required_number_of_votes) or \
        (votes[max_ind] >= len(class_votes)-2 and len(conf_subset[conf_subset > 0.9]) >= len(class_votes)-2 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco") or \
        (votes[max_ind] >= len(class_votes)-1 and len(conf_subset[conf_subset > 0.85]) >= len(class_votes)-1 and class_name[max_ind] != "Smiths" and class_name[max_ind] != "Costco"):  #testing this part
        
        final_results.append(class_name[max_ind])
        print(i,"TOCNO ", class_name[max_ind], "%d/%d" % (votes[max_ind], len(class_votes)), class_name[max_ind], conf_subset)#, res)
    else:
        print(i,"NETOCNO ", "Other", "%d/%d" % (votes[max_ind], len(class_votes)), class_name[max_ind], conf_subset)#, res)
        cnt_different += 1
        if votes[max_ind] == len(class_votes):
            cnt_same += 1
        final_results.append('Other')


print("Number of same predictions with low confidence labeled as other %d" % cnt_same)
sub = pd.DataFrame()
sub['Results'] = final_results
sub.to_csv('Mozgalo.csv', index=False, header=False)
print(len(final_results), "Others = %d" % cnt_different)
    