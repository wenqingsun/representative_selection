import glob, os
import preprocessing, postprocessing
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from time import time


# train_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_train_group/'
# test_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_test_group/'
train_dir = 'C:/experiments/temp_feature_vector/train/'
test_dir = 'C:/experiments/temp_feature_vector/test/'

out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/nn_dl1536_tsne_4/'

n_components = 4
perplexity = 30

def run():
    # get training set
    os.chdir(train_dir)
    folder_name = train_dir.split('/')[-2]
    # initiate train detections
    train_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.txt')[0:10]:
        with open(file) as f:
            lines = f.readlines()
            view_name = file.split('_feat.txt')[0]
            # adding detections from one view to this instance
            train_detections.update(lines, view_name)

    # get testing set
    os.chdir(test_dir)
    folder_name = test_dir.split('/')[-2]
    # initiate train detections
    test_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.txt')[0:1]:
        with open(file) as f:
            lines = f.readlines()
            view_name = file.split('_feat.txt')[0]
            # adding detections from one view to this instance
            test_detections.update(lines, view_name)

    # get train_x and train_y
    y = np.array(train_detections.get_all_tp_fp())
    X = np.array(train_detections.get_all_dl_feat_1536())
    del train_detections

    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100, 200]
    n_iters = [500, 1000, 5000]

    red = y == 0
    green = y == 1

    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    for i, n_iter in enumerate(n_iters):
        for j, perplexity in enumerate(perplexities):
            ax = subplots[i][j]

            t0 = time()
            tsne = manifold.TSNE(n_components=n_components, init='random',
                                 random_state=0, perplexity=perplexity, n_iter = n_iter)
            Y = tsne.fit_transform(X)
            t1 = time()
            print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
            ax.set_title("Perplexity=%d, iteration =%d" % (perplexity, n_iter))
            ax.scatter(Y[red, 0], Y[red, 1], c="r")
            ax.scatter(Y[green, 0], Y[green, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')

    plt.show()

    print(' ')

if __name__ == '__main__':
    run()
