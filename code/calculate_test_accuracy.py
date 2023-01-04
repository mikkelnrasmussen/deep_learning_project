import statistics 
import numpy as np
import scipy.stats as st

main_path = "classifier_results_"

acc_plus_list = []
for i in range(1,4):
    with open(main_path + "plus_VAE_" + str(i) + "/classifier_plus_VAE_" + str(i) + "_test_accuracy.txt") as f:
        acc = f.readlines()
        acc_plus_list.append(float(acc[0]))

print("Mean accuracy of VAE+: %.2f " %(statistics.mean(acc_plus_list)))
print("Standard Deviation of the accuracy is %.4f "%(statistics.stdev(acc_plus_list)))
conf_int_plus = st.t.interval(alpha=0.95, df=len(acc_plus_list)-1, loc=np.mean(acc_plus_list), scale=st.sem(acc_plus_list))
print(conf_int_plus)

acc_vanilla_list = []
for i in range(1,4):
    with open(main_path + "vanilla_VAE_" + str(i) + "/classifier_vanilla_VAE_" + str(i) + "_test_accuracy.txt") as f:
        acc = f.readlines()
        acc_vanilla_list.append(float(acc[0]))

print("Mean accuracy of vanilla VAE: %.2f " %(statistics.mean(acc_vanilla_list))) 
print("Standard Deviation of the sample is %.4f "%(statistics.stdev(acc_vanilla_list)))
conf_int_vanilla = st.t.interval(alpha=0.95, df=len(acc_vanilla_list)-1, loc=np.mean(acc_vanilla_list), scale=st.sem(acc_vanilla_list))
print(conf_int_vanilla)


#perform two sample t-test with equal variances
print(st.ttest_ind(a=acc_plus_list, b=acc_vanilla_list, equal_var=True))

acc_plain_list = []
for i in range(1,4):
    with open(main_path + "plain_" + str(i) + "/classifier_plain_" + str(i) + "_test_accuracy.txt") as f:
        acc = f.readlines()
        acc_plain_list.append(float(acc[0]))

print("Mean accuracy of plain classifier: %.2f " %(statistics.mean(acc_plain_list)))
print("Standard Deviation of the accuracy is %.4f "%(statistics.stdev(acc_plain_list)))
conf_int_plain = st.t.interval(alpha=0.95, df=len(acc_plain_list)-1, loc=np.mean(acc_plain_list), scale=st.sem(acc_plain_list))
print(conf_int_plain)

#perform two sample t-test with equal variances
print(st.ttest_ind(a=acc_plus_list, b=acc_plain_list, equal_var=True))