import os
import numpy as np
import pandas as pd

"""
B30_full_1mm_path = "./Training Set/1mm B30/full_1mm"
B30_quarter_1mm_path = "Training Set/1mm B30/quarter_1mm"
D45_full_1mm_path = "Training Set/1mm D45/full_1mm_sharp"
D45_quarter_1mm_path = "Training Set/1mm D45/quarter_1mm_sharp"

print(os.listdir(B30_full_1mm_path))

B30_full_1mm_paths = [os.path.join(B30_full_1mm_path, img_dir) for img_dir in os.listdir(B30_full_1mm_path)]
B30_quarter_1mm_paths = [os.path.join(B30_quarter_1mm_path, img_dir) for img_dir in os.listdir(B30_quarter_1mm_path)]
D45_full_1mm_paths = [os.path.join(D45_full_1mm_path, img_dir) for img_dir in os.listdir(D45_full_1mm_path)]
D45_quarter_1mm_paths = [os.path.join(D45_quarter_1mm_path, img_dir) for img_dir in os.listdir(D45_quarter_1mm_path)]

print(B30_full_1mm_paths)

B30_full_1mm = []
for img_dir in B30_full_1mm_paths:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        B30_full_1mm.append([file_path])
print(len(B30_full_1mm))

B30_quarter_1mm = []
for img_dir in B30_quarter_1mm_paths:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        B30_quarter_1mm.append([file_path])
print(len(B30_quarter_1mm))

D45_full_1mm = []
for img_dir in D45_full_1mm_paths:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        D45_full_1mm.append([file_path])
print(len(D45_full_1mm))

D45_quarter_1mm = []
for img_dir in D45_quarter_1mm_paths:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        D45_quarter_1mm.append([file_path])
print(len(D45_quarter_1mm))


B30_full_1mm = np.array(B30_full_1mm)
B30_quarter_1mm = np.array(B30_quarter_1mm)
D45_full_1mm = np.array(D45_full_1mm)
D45_quarter_1mm = np.array(D45_quarter_1mm)


ones = np.array([[1] for i in range(len(B30_full_1mm))])
quarters = np.array([[0.25] for i in range(len(B30_full_1mm))])
zeros = np.array([[0] for i in range(len(B30_full_1mm))])

B30_B30_full_quarter_1mm = np.concatenate((B30_full_1mm, B30_quarter_1mm, quarters), axis=1)
B30_D45_full_full_1mm = np.concatenate((B30_full_1mm, D45_full_1mm, ones), axis=1)
B30_D45_full_quarter_1mm = np.concatenate((B30_full_1mm, D45_quarter_1mm, quarters), axis=1) 
D45_D45_full_quarter_1mm = np.concatenate((D45_full_1mm, D45_quarter_1mm, quarters), axis=1) 
D45_B30_quarter_quarter_1mm = np.concatenate((D45_quarter_1mm, B30_quarter_1mm, zeros), axis=1) 
#D45_B30_full_quarter = np.concatenate((D45_full_1mm, B30_quarter_1mm, quarters), axis=1)


# Juntamos todas en una sola tabla
training_data = np.concatenate((B30_D45_full_full_1mm,
                                B30_B30_full_quarter_1mm, 
                                B30_D45_full_quarter_1mm,
                                D45_D45_full_quarter_1mm,
                                D45_B30_quarter_quarter_1mm))

# Convertimos en csv
df = pd.DataFrame(training_data)
df.to_csv('training_data.csv', index=False, sep=",", header=False)
"""

B30_quarter_1mm_path_test = "./Test Set/1mm B30/QD_1mm"
B30_quarter_3mm_path_test = "./Test Set/3mm B30/QD_3mm"
D45_quarter_1mm_path_test = "./Test Set/1mm D45/QD_1mm_sharp"
D45_quarter_3mm_path_test = "./Test Set/3mm D45/QD_3mm_sharp"

B30_quarter_1mm_paths_test  = [os.path.join(B30_quarter_1mm_path_test, img_dir,"quarter_1mm") for img_dir in os.listdir(B30_quarter_1mm_path_test)]
B30_quarter_3mm_paths_test  = [os.path.join(B30_quarter_3mm_path_test, img_dir,"quarter_3mm") for img_dir in os.listdir(B30_quarter_3mm_path_test)]
D45_quarter_1mm_paths_test  = [os.path.join(D45_quarter_1mm_path_test, img_dir,"quarter_1mm_sharp") for img_dir in os.listdir(D45_quarter_1mm_path_test)]
D45_quarter_3mm_paths_test  = [os.path.join(D45_quarter_3mm_path_test, img_dir,"quarter_3mm_sharp") for img_dir in os.listdir(D45_quarter_3mm_path_test)]

B30_quarter_1mm_test = []
for img_dir in B30_quarter_1mm_paths_test:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        B30_quarter_1mm_test.append([file_path])
print(len(B30_quarter_1mm_test))
print(B30_quarter_1mm_test[0])

B30_quarter_3mm_test = []
for img_dir in B30_quarter_3mm_paths_test:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        B30_quarter_3mm_test.append([file_path])
print(len(B30_quarter_3mm_test))
print(B30_quarter_3mm_test[0])

D45_quarter_1mm_test = []
for img_dir in D45_quarter_1mm_paths_test:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        D45_quarter_1mm_test.append([file_path])
print(len(D45_quarter_1mm_test))
print(D45_quarter_1mm_test[0])

D45_quarter_3mm_test = []
for img_dir in D45_quarter_3mm_paths_test:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        D45_quarter_3mm_test.append([file_path])
print(len(D45_quarter_3mm_test))
print(D45_quarter_3mm_test[0])

B30_quarter_1mm_test = np.array(B30_quarter_1mm_test)
B30_quarter_3mm_test = np.array(B30_quarter_3mm_test)
D45_quarter_1mm_test = np.array(D45_quarter_1mm_test)
D45_quarter_3mm_test = np.array(D45_quarter_3mm_test)

zeros_1mm = np.array([[0] for i in range(len(B30_quarter_1mm_test))])
zeros_3mm = np.array([[0] for i in range(len(B30_quarter_3mm_test))])

B30_D45_quarter_quarter_1mm_1mm_test = np.concatenate((B30_quarter_1mm_test, D45_quarter_1mm_test, zeros_1mm), axis=1)
B30_D45_quarter_quarter_3mm_3mm_test = np.concatenate((B30_quarter_3mm_test, D45_quarter_3mm_test, zeros_3mm), axis=1)

test_data = np.concatenate((B30_D45_quarter_quarter_1mm_1mm_test,
                            B30_D45_quarter_quarter_3mm_3mm_test))

# Convertimos en csv
df_test = pd.DataFrame(test_data)
df_test.to_csv('test_data.csv', index=False, sep=",", header=False)