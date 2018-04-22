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
cnt_different = 0
confidence_threshold = 0.95
insecure_thresh = 0.7
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
    max_ind = np.argmax(votes) #imam samo 2 za ansamblat pa ovo daje krivu stvar
    max_class_name = class_name[max_ind]
    if len(votes) == 2:
        max_ind = 0 if confidence[0] > confidence[1] else 1
        max_class_name = class_votes[max_ind]

    max_vote_indices = np.where(class_votes==max_class_name)
    other_vote_indices = np.where(class_votes!=max_class_name)
    conf_subset = confidence[max_vote_indices]
    conf_oposite_subset = confidence[other_vote_indices]
    # if votes[max_ind] == len(class_votes) and len(conf_subset[conf_subset > confidence_threshold]) >= np.ceil(len(class_votes)/2): 
    if votes[max_ind] >= np.ceil(len(class_votes)/2) and len(conf_subset[conf_subset > confidence_threshold]) >= np.ceil(len(class_votes)/2) and len(conf_oposite_subset[conf_oposite_subset > insecure_thresh]) < np.ceil(len(class_votes)/2): 
        final_results.append(max_class_name)
        print(i,"NETOCNO ", "Other", res)
    else:
        cnt_different += 1
        final_results.append('Other')

sub = pd.DataFrame()
sub['Results'] = final_results
sub.to_csv('Mozgalo.csv', index=False, header=False)
print(len(final_results), "Others = %d" % cnt_different)
    