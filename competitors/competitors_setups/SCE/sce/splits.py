import numpy as np

fout = open('data/citeseer_labels.txt', 'r')
label_file = fout.readlines()

#np.random.seed(123)
label_dict = {}
for line in label_file:
    if line.strip('\n'):
        line = line.strip('\n').split(' ')
        if line[1] not in label_dict:
            label_dict[line[1]] = []
            label_dict[line[1]].append(line[0])
        else:
            label_dict[line[1]].append(line[0])
fout.close()

train_file = open('train_text.txt',  'w')
val_file = open('val_text.txt', 'w')
test_file = open('test_text.txt', 'w')

#The size of training set are [5, 20] per class for each dataset respectively.
for i in range(50):
    idx_train = []
    idx_val = []
    idx_test = []
    for j in label_dict:
        if len(label_dict[j]) > 20:
            train = list(np.random.choice(label_dict[j], size=5, replace=False))
            val = list(np.random.choice(list(set(label_dict[j]) - set(train)), size=5, replace=False))
            test = list(set(label_dict[j]) - set(train) - set(val))
        else:
            train = list(np.random.choice(label_dict[j], size=int(len(label_dict[j]) * 0.5), replace=False))
            test = list(set(label_dict[j]) - set(train))
        idx_train.extend([int(x) for x in train])
        idx_val.extend([int(x) for x in val])
        idx_test.extend([int(x) for x in test])
        # val = list(np.random.choice(list(set(label_dict[j]) - set(train)), size=num, replace=False))
    train_file.write(str(idx_train) + '\n')
    val_file.write(str(idx_val) + '\n')
    test_file.write(str(idx_test) + '\n')

train_file.close()
val_file.close()
test_file.close()
