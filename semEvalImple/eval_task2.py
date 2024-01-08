import sys
import numpy as np
import sys
def pp(*p): # for debug
    for idx, arg_value in enumerate(p): print(f'{idx}:	val= {arg_value}	type= {type(arg_value)}')
    sys.exit()

def format_judge(submission):
    '''
    judge if the submission file's format is legal
    :param submission: submission file
    :return: False for illegal
             True for legal
    '''
    # submission: [sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid]

    if submission[1] == '-1' or submission[2] == '-1':
        return False
    if (submission[3] == '-1' and submission[4] != '-1') or (submission[3] != '-1' and submission[4] == '-1'):
        return False
    if (int(submission[1]) >= int(submission[2])) or (int(submission[3]) > int(submission[4])):
        return False
    if not (int(submission[1]) >= -1 and int(submission[2]) >= -1 and int(submission[3]) >= -1 and int(submission[4]) >= -1):
        return False
    return True
def get_inter_id(submission_idx, truth_idx):
    # print(submission_idx)
    # print(truth_idx)
    sub_start = int(submission_idx[0])
    sub_end = int(submission_idx[1])
    truth_start = int(truth_idx[0])
    truth_end = int(truth_idx[1])
    if sub_end < truth_start or sub_start > truth_end:
        return False, -1, -1
    return True, max(sub_start, truth_start), min(sub_end, truth_end)    
def metrics_task2(submission_list, truth_list):
    # submission_list:  [[sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid], ...]
    # truth_list:       [[sentenceID,sentence, antecedent_startid,antecedent_endid,consequent_startid,consequent_endid], ...]

    # pp(submission_list[0], truth_list[0])
    f1_score_all = []
    precision_all = []
    recall_all = []

    for i in range(len(submission_list)):
        assert submission_list[i][0] == truth_list[i][0]
        submission = submission_list[i]
        truth = truth_list[i]
        precision = 0
        recall = 0
        f1_score = 0

        if format_judge(submission):
            # truth processing
            sentence = truth[1]

            t_a_s = int(truth[2])       # truth_antecedent_startid
            t_a_e = int(truth[3])       # truth_antecedent_endid
            t_c_s = int(truth[4])       # truth_consequent_startid
            t_c_e = int(truth[5])       # truth_consequent_endid

            s_a_s = int(submission[1])  # submission_antecedent_startid
            s_a_e = int(submission[2])  # submission_antecedent_endid
            s_c_s = int(submission[3])  # submission_consequent_startid
            s_c_e = int(submission[4])  # submission_consequent_endid

            truth_ante_len = len(sentence[t_a_s : t_a_e].split())
            if truth[4] == '-1':
                truth_cons_len = 0
            else:
                truth_cons_len = len(sentence[t_c_s : t_c_e].split())
            truth_len = truth_ante_len + truth_cons_len

            # submission processing
            submission_ante_len = len(sentence[s_a_s : s_a_e].split())
            if submission[3] == '-1':
                submission_cons_len = 0
            else:
                submission_cons_len = len(sentence[s_c_s : s_c_e].split())
            submission_len = submission_ante_len + submission_cons_len

            # intersection
            inter_ante_flag, inter_ante_startid, inter_ante_endid = get_inter_id([s_a_s, s_a_e], [t_a_s, t_a_e])
            if truth_cons_len == 0 or submission_cons_len == 0:
                inter_cons_startid = 0
                inter_cons_endid = 0
                inter_cons_flag = False
            else:
                inter_cons_flag, inter_cons_startid, inter_cons_endid = get_inter_id([s_c_s, s_c_e], [t_c_s, t_c_e])

            inter_ante_len = 0
            inter_cons_len = 0
            if inter_ante_flag:
                inter_ante_len = len(sentence[inter_ante_startid : inter_ante_endid].split())
            if inter_cons_flag:
                inter_cons_len = len(sentence[inter_cons_startid : inter_cons_endid].split())
            inter_len = inter_ante_len + inter_cons_len

            # calculate precision, recall, f1-score
            if inter_len > 0:
                precision = inter_len / submission_len
                recall = inter_len / truth_len
                f1_score = 2 * precision * recall / (precision + recall)

        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)

    f1_mean = np.mean(f1_score_all)
    precision_mean = np.mean(precision_all)
    recall_mean = np.mean(recall_all)
    return f1_mean, precision_mean, recall_mean,f1_score_all

def evaluate2(truth_reader, submission_list, true_sentence):
    truth_list=[]
    not_em = 0
    for idx, line in enumerate(truth_reader):
        tmp = []
        submission_line = submission_list[idx]
        if line[0] != submission_line[0]:
            # print("the sentence id is not matched")
            sys.exit("Sorry, the sentence id is not matched.")
        tmp.append(line[0])    # sentenceID
        tmp.append(true_sentence[idx][1])
        tmp.extend(line[-4:])  # ante_start, ante_end, conq_start, conq_end
        truth_list.append(tmp)

        if submission_line[1] != tmp[2] or submission_line[2] != tmp[3] or submission_line[3] != tmp[4] or submission_line[4] != tmp[5]:
            not_em += 1
    if len(truth_list) != len(submission_list):
        # print("please check the rows#")
        sys.exit("Please check the number of rows in your .csv file! It should consistent with 'train.csv' in practice stage, and should be consistent with 'test.csv' in evaluation stage.")

    exact_match = (len(truth_list) - not_em) / len(truth_list)
    f1_mean, recall_mean, precision_mean,f1_score_all = metrics_task2(submission_list, truth_list)

    return f1_mean, recall_mean, precision_mean, exact_match,f1_score_all


import csv

def extract_base_sentence_id(sentence_id):
    return sentence_id[:6]

# Read test.csv
with open('test.csv', 'r',encoding='utf-8') as test_file:
    test_reader = csv.reader(test_file, delimiter=',')
    next(test_reader)  # Skip header
    test_data = [row for row in test_reader]

# Read antecedents.tsv
with open('antecedents.tsv', 'r',encoding='utf-8') as antecedents_file:
    antecedents_reader = csv.reader(antecedents_file, delimiter='\t')
    antecedents_data = [row for row in antecedents_reader]

# Read consequent.tsv
with open('consequents.tsv', 'r',encoding='utf-8') as consequent_file:
    consequent_reader = csv.reader(consequent_file, delimiter='\t')
    consequent_data = [row for row in consequent_reader]

# Extract relevant data for evaluate2 function
submission_list = []
true_sentence = []
for i in range(len(test_data)):
    base_sentence_id = extract_base_sentence_id(test_data[i][0])

    # Find corresponding antecedent and consequent data
    antecedent_entry = [row for row in antecedents_data if row[0].startswith(base_sentence_id+"A0")]
    consequent_entry = [row for row in consequent_data if row[0].startswith(base_sentence_id+"C0")]

    # Extract necessary fields
    if len(antecedent_entry) != 0:
        antecedent_start_id = antecedent_entry[0][3]
        antecedent_end_id = antecedent_entry[0][4]
    else:
        antecedent_start_id = -1
        antecedent_end_id = -1
    if len(consequent_entry) != 0:
        consequent_start_id = consequent_entry[0][3]
        consequent_end_id = consequent_entry[0][4]
    else:
        consequent_start_id = -1
        consequent_end_id = -1
    
    # Create submission entry
    submission_entry = [test_data[i][0],  antecedent_start_id, antecedent_end_id, consequent_start_id, consequent_end_id]
    submission_list.append(submission_entry)
    
    # Create true sentence entry
    true_sentence_entry = [test_data[i][0], test_data[i][1]]
    true_sentence.append(true_sentence_entry)

    
# Call evaluate2 function
f1_mean, recall_mean, precision_mean, exact_match, f1_score_all = evaluate2(test_data, submission_list, true_sentence)

# Print the results
print("F1 Mean:", f1_mean)
print("Recall Mean:", recall_mean)
print("Precision Mean:", precision_mean)
print("Exact Match:", exact_match)
# print("F1 Score All:", f1_score_all)