from sklearn import metrics
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_risk_coverage_info(prob_list, em_list):
    num1 = int(len(prob_list)/2)
    num2 = len(prob_list) - num1
    sources = [0 for i in range(num1)]
    sources.extend([1 for i in range(num2)])
    assert len(sources) == len(prob_list)
    tuples = [(x,y,z) for x,y,z in zip(prob_list, em_list, sources)]
    sorted_tuples = sorted(tuples, key=lambda x: -x[0])
    sorted_probs = [x[0] for x in sorted_tuples]
    sorted_em = [x[1] for x in sorted_tuples]
    sorted_sources = [x[2] for x in sorted_tuples]
    total_questions = len(sorted_em)
    total_correct = 0
    covered = 0
    risks = []
    coverages = []

    for em, prob in zip(sorted_em, sorted_probs):
        covered += 1
        if em:
            total_correct += 1
        risks.append(1 - (total_correct/covered))
        coverages.append(covered/total_questions)        
    auc = round(metrics.auc(coverages, risks), 4)

    
    return risks, coverages, auc, sorted_sources, sorted_em, sorted_probs

def get_coverage_cutoff(risks, accuracy_cutoff):
    index = len(risks)
    while risks[index-1] >= (1.0-accuracy_cutoff) and index > 0:
        index -= 1
    return index

def auc_show(probs, correct, plot_graph=False):

    all_risks, all_coverages, all_aucs = [], [], []
    all_sorted_sources, all_sorted_em = [], []

    risks, coverages, auc, sorted_sources, sorted_em, sorted_probs = get_risk_coverage_info(probs, correct)

    all_risks.append(risks)
    all_coverages.append(coverages)
    all_aucs.append(auc)
    all_sorted_sources.append(sorted_sources)
    all_sorted_em.append(sorted_em)
    
    avg_risks = np.mean(all_risks, axis=0)
    avg_coverages = np.mean(all_coverages, axis=0)
    
    avg_auc = np.mean(all_aucs)

    

    values = list(np.arange(99.5, 1, -0.5))
    plot_values = {
        'acc': [],
        'covs': [],
        'probs': []
    }
    for i in values:
        index = get_coverage_cutoff(avg_risks, i/100) - 1
        if index==-1:
            continue
        cov = round((100 * avg_coverages[index]), 4)
        prob = sorted_probs[index]
        plot_values['acc'].append(i)
        plot_values['covs'].append(cov)
        plot_values['probs'].append(prob)
        # print("{accuracy}, {coverage}, {prob} ".format(coverage=cov, accuracy =i, prob=round(prob,4))) 
        if(cov == 100):
            break

    if(plot_graph):
        plt.plot(coverages,risks)
        plt.show()
        
    return round(100*avg_auc, 2), coverages, risks, plot_values

def get_auc(df, plot_graph=False):
    accuracy = round(100*df["correct"].mean(),2)
    print("Accuracy ", round(100*df["correct"].mean(),2))
    achieved_auc, coverages, risks, plot_values = auc_show(list(df["maxProb"]), list(df["correct"]), plot_graph=plot_graph)
    print("Achieved AUC: ", achieved_auc)
    
    dev_len = df.shape[0] + 1
    ideal_probs_list = np.asarray(range(dev_len)[1:])/dev_len 
    em_list = df["correct"]
    em_list = np.sort(em_list)
    min_auc, min_coverages, min_risks, _ = auc_show(list(ideal_probs_list), list(em_list))
    print("Minimum Possible AUC: ",min_auc)
    
    return (accuracy, achieved_auc, min_auc, coverages, risks, plot_values)

if __name__=="__main__":
    hetero_prediction_files = ["snli_eval_prob_n_preds_hetero.json", "swag_eval_prob_n_preds_hetero.json", "csqa_eval_prob_n_preds_hetero.json", "anli_eval_prob_n_preds_hetero.json", "siqa_eval_prob_n_preds_hetero.json"]
    homo_prediction_files = ["snli_eval_prob_n_preds_homo.json", "swag_eval_prob_n_preds_homo.json", "csqa_eval_prob_n_preds_homo.json", "anli_eval_prob_n_preds_homo.json", "siqa_eval_prob_n_preds_homo.json"]
    
    # print(df1.head())

    dataset_name = {
        "snli": "SNLI",
        "swag": "SWAG",
        "csqa": "Commonsense QA",
        "anli": "Abductive NLI",
        "siqa": "Social IQA"
    }
    
    for i in range(5):
        df1 = pd.read_json(hetero_prediction_files[i])
        df2 = pd.read_json(homo_prediction_files[i])
        ds_name = dataset_name[hetero_prediction_files[i].split('_')[0]]
        print("\n"+ ds_name + " - Heterogenous metrics\n")
        accuracy_hetero, achieved_auc_hetero, min_auc_hetero, coverages_hetero, risks_hetero, plot_values_hetero = get_auc(df1,False)
        print("\n"+ ds_name + " - Homogenous metrics\n")
        accuracy_homo, achieved_auc_homo, min_auc_homo, coverages_homo, risks_homo, plot_values_homo = get_auc(df2,False)

        legend = ["Heterogenous Sampling", "Homogenous Sampling"]


        t_v_title = "Threshold vs Accuracy for " + dataset_name[hetero_prediction_files[i].split('_')[0]]
        plt.subplot(2,2,1)
        plt.xlabel("Probability (Threshold)")
        plt.ylabel("Accuracy (%)")
        plt.title(t_v_title)
        plt.plot(plot_values_hetero['probs'], plot_values_hetero['acc'])
        plt.plot(plot_values_homo['probs'], plot_values_homo['acc'])
        plt.legend(legend)
        # plt.savefig(t_v_title.replace(' ', '_') + ".png", dpi=300)
        # plt.show()
        # plt.clf()

        plt.subplot(2,2,2)
        c_a_title = "Coverage vs Accuracy for " + dataset_name[hetero_prediction_files[i].split('_')[0]]
        plt.xlabel("Coverage")
        plt.ylabel("Accuracy (%)")
        plt.title(c_a_title)
        plt.plot(plot_values_hetero['covs'], plot_values_hetero['acc'])
        plt.plot(plot_values_homo['covs'], plot_values_homo['acc'])
        plt.legend(legend)
        # plt.savefig(c_a_title.replace(' ', '_') + ".png", dpi=300)
        # plt.show()
        # plt.clf()

        plt.subplot(2,2,3)
        t_c_title = "Threshold vs Coverage for " + dataset_name[hetero_prediction_files[i].split('_')[0]]
        plt.xlabel("Probability (Threshold)")
        plt.ylabel("Coverage")
        plt.title(t_c_title)
        plt.plot(plot_values_hetero['probs'], plot_values_hetero['covs'])
        plt.plot(plot_values_homo['probs'], plot_values_homo['covs'])   
        plt.legend(legend)
        # plt.savefig(t_c_title.replace(' ', '_') + ".png", dpi=300)
        # plt.show()
        # plt.clf()


        plt.subplot(2,2,4)
        c_r_title = "Coverage vs Risk for " + dataset_name[hetero_prediction_files[i].split('_')[0]]
        plt.xlabel("Coverage")
        plt.ylabel("Risk")
        plt.title(c_r_title)
        plt.plot(coverages_hetero, risks_hetero)
        plt.plot(coverages_homo, risks_homo)   
        plt.legend(legend)
        # plt.savefig(c_r_title.replace(' ', '_') + ".png", dpi=300)
        # plt.show()
        # plt.clf()

        # plt.savefig(ds_name+'_metrics_graph.png')
        plt.show()
        





