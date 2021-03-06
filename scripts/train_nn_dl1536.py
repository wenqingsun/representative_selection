import glob, os
import preprocessing, postprocessing
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE


from sklearn.preprocessing import StandardScaler

# train_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_train_group/'
# test_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_test_group/'
train_dir = 'C:/experiments/temp_feature_vector/train/'
test_dir = 'C:/experiments/temp_feature_vector/test/'

out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/nn_dl1536_tsne_4/'

n_components = 4
perplexity = 100

def run():
    # get training set
    os.chdir(train_dir)
    folder_name = train_dir.split('/')[-2]
    # initiate train detections
    train_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.txt')[0:300]:
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
    train_y = np.array(train_detections.get_all_tp_fp())
    train_x = np.array(train_detections.get_all_dl_feat_1536())
    del train_detections
    tsne = TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    train_x = tsne.fit_transform(train_x)

    # get test_x and test_y
    test_y = np.array(test_detections.get_all_tp_fp())
    test_x = np.array(test_detections.get_all_dl_feat_1536())
    del test_detections
    test_x = tsne.fit_transform(test_x)

    # scale the features
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # start the training
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    mlp.fit(train_x, train_y)

    # predictions on train set
    predictions_train = mlp.predict(train_x)
    prob_train = mlp.predict_proba(train_x)[:,1]
    print('This is the training set result: \n')
    print(confusion_matrix(train_y, predictions_train))
    print(classification_report(train_y, predictions_train))

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(train_y, prob_train, pos_label=1)
    az_train = auc(fpr, tpr)
    print('AUC for train set is ' + str(az_train))

    # predictions on test set
    predictions_test = mlp.predict(test_x)
    prob_test = mlp.predict_proba(test_x)[:,1]
    print('This is the testing set result: \n')
    print(confusion_matrix(test_y, predictions_test))
    print(classification_report(test_y, predictions_test))

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(test_y, prob_test, pos_label=1)
    az_test = auc(fpr, tpr)
    print('AUC for testing set is ' + str(az_test))

    # record the predictions into the class
    # train_detections.record_scores(prob_train)
    # test_detections.record_scores(prob_test)

    # Now test on each view and dump txt for scoring
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        folder_name = 'test'
        test_detections_one_view = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
            test_detections_one_view.update(lines, view_name)

        # Get the feature values
        test_x = np.array(test_detections_one_view.get_all_dl_feat_1536())
        test_x = tsne.fit_transform(test_x)
        test_x = scaler.transform(test_x)
        predictions_test_one_view = mlp.predict_proba(test_x)[:,1]

        # record the predictions
        test_detections_one_view.record_scores(predictions_test_one_view)

        postprocessing.pick_one_mark_one_group_use_prediction(test_detections_one_view)
        postprocessing.save_to_txt(test_detections_one_view, out_dir)


if __name__ == '__main__':
    run()