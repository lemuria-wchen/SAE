# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:40:34 2021

@author: LYZ
"""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# accuracy, precision, recall, f1
def measures(preds, labels):
    p, r, f1 = {}, {}, {}
    
    acc = accuracy_score(labels, preds)
    
    p["avg"] = precision_score(labels, preds, average="macro")   
    r["avg"] = recall_score(labels, preds, average="macro")    
    f1["avg"] = f1_score(labels, preds, average="macro")
    
    p["all"] = precision_score(labels, preds, average=None)   
    r["all"] = recall_score(labels, preds, average=None)    
    f1["all"] = f1_score(labels, preds, average=None)
    
    return acc, p, r, f1

def transfer_measures(transfer_type, old_preds, labels):
    preds = []
    # 五分类模型转移到二分类上
    if transfer_type == "5-2":
        for i in range(len(old_preds)):
            if old_preds[i] == 3: preds.append(1)
            else: preds.append(0)
    # 二分类模型转移到五分类上
    elif transfer_type == "2-5":
        preds = old_preds
        
    p, r, f1 = {}, {}, {}
    
    acc = accuracy_score(labels, preds)
    
    p["avg"] = precision_score(labels, preds, average="macro")   
    r["avg"] = recall_score(labels, preds, average="macro")    
    f1["avg"] = f1_score(labels, preds, average="macro")
    
    p["all"] = precision_score(labels, preds, average=None)   
    r["all"] = recall_score(labels, preds, average=None)    
    f1["all"] = f1_score(labels, preds, average=None)
    
    return acc, p, r, f1
    

# print report
def report_log(report_type, loss, acc, p, r, f1, log_file, args):
    # report_type: "Train", "Dev", "Test"
    report1 = "\t" + report_type + " Measures Report"
    report2 = "\t\tLoss: {:.3f}".format(loss)
    report3 = "\t\tAccuracy: {:.2f}%".format(acc*100)
    
    p_avg, r_avg, f1_avg = p["avg"]*100, r["avg"]*100, f1["avg"]*100
    report4 = f"\t\tPrecision: {p_avg:.2f}% | "
    report5 = f"\t\tRecall: {r_avg:.2f}% | "
    report6 = f"\t\tF1-Score: {f1_avg:.2f}% | "
    
    for i in range(args.label_dim):
        p_all, r_all, f1_all = p["all"][i]*100, r["all"][i]*100, f1["all"][i]*100
        report4 += f"{p_all:.2f}% "
        report5 += f"{r_all:.2f}% "
        report6 += f"{f1_all:.2f}% "
        
    report4 = report4
    report5 = report5
    report6 = report6
    
    report = report1 + "\n" + report2 + "\n" + report3 + "\n" + \
            report4 + "\n" + report5 + "\n" + report6 + "\n"
            
    print(report)
    log_file.write(report)
    
# print report
def transfer_report_log(report_type, loss, acc, p, r, f1, log_file, args):
    # report_type: "Train", "Dev", "Test"
    report1 = "\t" + report_type + " Measures Report"
    report2 = "\t\tLoss: {:.3f}".format(loss)
    report3 = "\t\tAccuracy: {:.2f}%".format(acc*100)
    
    p_avg, r_avg, f1_avg = p["avg"]*100, r["avg"]*100, f1["avg"]*100
    report4 = f"\t\tPrecision: {p_avg:.2f}% | "
    report5 = f"\t\tRecall: {r_avg:.2f}% | "
    report6 = f"\t\tF1-Score: {f1_avg:.2f}% | "
    
    for i in range(args.transfer_label_dim):
        p_all, r_all, f1_all = p["all"][i]*100, r["all"][i]*100, f1["all"][i]*100
        report4 += f"{p_all:.2f}% "
        report5 += f"{r_all:.2f}% "
        report6 += f"{f1_all:.2f}% "
        
    report4 = report4
    report5 = report5
    report6 = report6
    
    report = report1 + "\n" + report2 + "\n" + report3 + "\n" + \
            report4 + "\n" + report5 + "\n" + report6 + "\n"
            
    print(report)
    log_file.write(report)