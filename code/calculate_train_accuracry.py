import statistics 
import numpy as np
import scipy.stats as st

main_path = "classifier_results_"

acc_plus_list = []
for i in range(1,4):
    acc_plus = np.load(main_path + "plus_VAE_" + str(i) + "/train_accuracies.npy")
    acc_plus_list.append(acc_plus)
print(max(acc_plus_list[0]))
print(max(acc_plus_list[1]))
print(max(acc_plus_list[2]))

max_train_plus_acc = [max(acc_plus_list[0]), 
                      max(acc_plus_list[1]),
                      max(acc_plus_list[2])]

print("Mean accuracy of VAE+: %.2f " %(statistics.mean(max_train_plus_acc)))
print("Standard Deviation of the accuracy is %.4f "%(statistics.stdev(max_train_plus_acc)))
conf_int_plus = st.t.interval(alpha=0.95, 
                              df=len(max_train_plus_acc)-1, 
                              loc=np.mean(max_train_plus_acc), 
                              scale=st.sem(max_train_plus_acc))
print(conf_int_plus)

acc_vanilla_list = []
for i in range(1,4):
    acc_plus = np.load(main_path + "vanilla_VAE_" + str(i) + "/train_accuracies.npy")
    acc_vanilla_list.append(acc_plus)


print(max(acc_vanilla_list[0]))
print(max(acc_vanilla_list[1]))
print(max(acc_vanilla_list[2]))

max_train_vanilla_acc = [max(acc_vanilla_list[0]), 
                         max(acc_vanilla_list[1]),
                         max(acc_vanilla_list[2])]

print("Mean accuracy of vanilla VAE: %.2f " %(statistics.mean(max_train_vanilla_acc))) 
print("Standard Deviation of the sample is %.4f "%(statistics.stdev(max_train_vanilla_acc)))
conf_int_vanilla = st.t.interval(alpha=0.95, 
                                 df=len(max_train_vanilla_acc)-1, 
                                 loc=np.mean(max_train_vanilla_acc), 
                                 scale=st.sem(max_train_vanilla_acc))
print(conf_int_vanilla)

#perform two sample t-test with equal variances
print(st.ttest_ind(a=max_train_plus_acc, b=max_train_vanilla_acc, equal_var=True))

acc_plain_list = []
for i in range(1,4):
    acc_plain = np.load(main_path + "plain_" + str(i) + "/train_accuracies.npy")
    acc_plain_list.append(acc_plain)


print(max(acc_plain_list[0]))
print(max(acc_plain_list[1]))
print(max(acc_plain_list[2]))

max_train_plain_acc = [max(acc_plain_list[0]), 
                         max(acc_plain_list[1]),
                         max(acc_plain_list[2])]

print("Mean accuracy of vanilla VAE: %.2f " %(statistics.mean(max_train_plain_acc))) 
print("Standard Deviation of the sample is %.4f "%(statistics.stdev(max_train_plain_acc)))
conf_int_vanilla = st.t.interval(alpha=0.95, 
                                 df=len(max_train_plain_acc)-1, 
                                 loc=np.mean(max_train_plain_acc), 
                                 scale=st.sem(max_train_plain_acc))
print(max_train_plain_acc)

#perform two sample t-test with equal variances
print(st.ttest_ind(a=max_train_plus_acc, b=max_train_plain_acc, equal_var=True))