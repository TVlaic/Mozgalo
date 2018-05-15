import os
import pandas as pd
import numpy as np

num = 4
first = ['/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidenceCenterLoss0.901_score_0.5.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidence_center_loss1.0.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidence_third.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidenceFifth.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidenceSixth0.0191-0025.hdf5.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidenceInception.csv',
'/home/user/Mozgalo/outputs/0.94685Max/SubmissionWithConfidenceInceptionLastBatchNorm.csv'
]
second = ['/home/user/Mozgalo/scripts/IndividualResults/FirstResidualAttentionModel.csv',
'/home/user/Mozgalo/scripts/IndividualResults/SecondResidualAttentionModel.csv',
'/home/user/Mozgalo/scripts/IndividualResults/ThirdResidualAttentionModel.csv',
'/home/user/Mozgalo/scripts/IndividualResults/FifthResidualAttentionModel.csv',
'/home/user/Mozgalo/scripts/SubmissionWithConfidence.csv',
'/home/user/Mozgalo/scripts/IndividualResults/FirstInceptionModel.csv',
'/home/user/Mozgalo/scripts/IndividualResults/FirstInceptionModelBNorm.csv'
]

cnt_match = 0
cnt_match_other = 0
number_of_others = 0
cnt_prob = 0
changed_other = 0
changed_to_other = 0
with open('./current_best.csv', 'r') as f1:
    with open('./Mozgalo.csv', 'r') as f2:
# with open(first[num], 'r') as f1:
#     with open(second[num], 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for i, line in enumerate(lines1):
            str1 = lines1[i].strip().split(',')[0]
            str2 = lines2[i].strip().split(',')[0]
            # conf1 = float(lines1[i].strip().split(',')[1])
            # conf2 = float(lines2[i].strip().split(',')[1])
            # if conf < 0.95:
            #     str2 = "Other"
            if str1 == str2:
                if  str2 != "Other":
                    cnt_match += 1
                else:
                    cnt_match_other +=1 
            else:
                # if str1=='Other':
                #     print(i, lines1[i].strip(), " -> "lines2[i].strip())
                if lines1[i].strip() == "Other":
                    changed_other +=1
                if lines2[i].strip() == "Other":
                    changed_to_other +=1
                print(i, lines1[i].strip(), " -> ",lines2[i].strip())
                #     cnt_prob+=1
                # print("Vote %d" % i, lines1[i].strip(), lines2[i].strip())
            # if str2 == "Other":
            #     number_of_others += 1
# print("Total number of others = %d" %number_of_others)
print("Matching = %d" % cnt_match)#, "Matching with high probability = %d" % cnt_prob)
print("Matching Others = %d" % cnt_match_other, "Different with high probability = %d" % cnt_prob)#)
print("Total matching = %d" %(cnt_match_other+cnt_match))
print("Changed from other = %d" % changed_other)
print("Changed to other = %d" % changed_to_other)
    