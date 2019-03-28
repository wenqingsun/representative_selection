# Jing-Long generated the group txt files, and Wenqing generated feature extraction and logits txt files, but the order is different.
# This script was used to match them together by using x y z information, so the feature vector will have the label as well.

import os, glob

dir1 = 'C:/experiments/temp_feature_vector/TP_test/' # I temporarily put the feature vector files in my local drive
dir2 = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_test_group/'
out_dir = 'C:/experiments/temp_feature_vector/TP_test_combine/'

os.chdir(dir1)
for file1 in glob.glob('*.txt'):
    base = file1.split('_score')[0]
    file2 = os.path.join(dir2, base+ '_feat.txt')

    # Check the file exsit or not
    if (os.path.isfile(file2) == 0):
        print('Could not find matching file!')

    # Create matching list
    x2 = []
    y2 = []
    z2 = []
    with open(file2) as f2:
        lines2 = f2.readlines()
        output = []
        for line2 in lines2:
            x2.append(line2.split()[1])
            y2.append(line2.split()[2])
            z2.append(line2.split()[3])
            output.append([i for i in line2.split()])

    # matching the x y z
    with open(file1) as f1:
        lines1 = f1.readlines()
        for idx, line1 in enumerate(lines1):
            if idx % 5 == 0: # because of 5 scales
                x1 = line1.split()[2]
                y1 = line1.split()[3]
                z1 = line1.split()[4]
                for i in range(len(lines2)):
                    if x1 == x2[i] and y1 == y2[i] and z1 == z2[i]:
                        if len(output[i]) < 1547: # Since Jin-long original file has duplicated entries, use this to prevent over adding features
                            output[i].extend(line1.split()[7:])

    out_file = os.path.join(out_dir, base+ '_feat.txt')
    with open(out_file, 'w') as f_out:
        for i in range(len(output)):
            if (len(output[i]) != 1547):
                print('element length is not correct for case ' + base + ', expected 1546 actual get ' + str(len(output[i])))
            str_oneline = ' '.join(str(e) for e in output[i])
            f_out.write(str_oneline + '\n')
