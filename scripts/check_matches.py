import os
import pandas as pd
import numpy as np

cnt_match = 0
cnt_match_other = 0
number_of_others = 0
cnt_prob = 0
with open('./current_best.csv', 'r') as f1:
    with open('./Mozgalo.csv', 'r') as f2:
# with open('/home/user/Mozgalo/outputs/results_for_ansambling/SubmissionWithConfidence_third.csv', 'r') as f1:
#     with open('/home/user/Mozgalo/outputs/results_for_ansambling/SubmissionWithConfidenceFourth.csv', 'r') as f2:
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
                print(i, lines1[i].strip(), " -> ",lines2[i].strip())
                #     cnt_prob+=1
                # print("Vote %d" % i, lines1[i].strip(), lines2[i].strip())
            # if str2 == "Other":
            #     number_of_others += 1
# print("Total number of others = %d" %number_of_others)
print("Matching = %d" % cnt_match)#, "Matching with high probability = %d" % cnt_prob)
print("Matching Others = %d" % cnt_match_other, "Different with high probability = %d" % cnt_prob)#)
print("Total matching = %d" %(cnt_match_other+cnt_match))
    