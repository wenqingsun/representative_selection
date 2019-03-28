import glob, os
import preprocessing, postprocessing
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc
from sklearn.externals import joblib


from sklearn.preprocessing import StandardScaler

# train_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_train_group/'
# test_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_test_group/'
train_dir = 'E:/Perforce/DL/IP3D/dev/DeepLearning/common/object_detection_holx/Stage3/labeled_data/v6_5logits/lowres2_half01_grp/'
test_dir = 'E:/Perforce/DL/IP3D/dev/DeepLearning/common/object_detection_holx/Stage3/labeled_data/v6_5logits/lowres2_half01_grp/'

out_dir = 'C:/experiments/pick_marker/results/score_txt/nn_od_dl_logit/'

def run():
    # get training set
    os.chdir(train_dir)
    folder_name = train_dir.split('/')[-2]
    # initiate train detections
    train_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.feat'):
        with open(file) as f:
            lines = f.readlines()
            view_name = file.split('.feat')[0]
            # adding detections from one view to this instance
            train_detections.update(lines, view_name)#, update_tp_group_only = True)

    # get testing set
    os.chdir(test_dir)
    folder_name = test_dir.split('/')[-2]
    # initiate train detections
    test_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.feat'):
        with open(file) as f:
            lines = f.readlines()
            view_name = file.split('.feat')[0]
            # adding detections from one view to this instance
            test_detections.update(lines, view_name)

    # get train_x and train_y
    train_y = np.array(train_detections.get_all_tp_fp())
    train_x = np.array([train_detections.get_all_od(), train_detections.get_all_dl(), train_detections.get_all_logit(), train_detections.get_all_area()]).transpose()


    # get test_x and test_y
    test_y = np.array(test_detections.get_all_tp_fp())
    test_x = np.array([test_detections.get_all_od(), test_detections.get_all_dl(), test_detections.get_all_logit(), test_detections.get_all_area()]).transpose()

    # scale the features
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # start the training
    mlp = MLPClassifier(hidden_layer_sizes=(8,4))
    mlp.fit(train_x, train_y)

    # save the model
    joblib.dump(mlp, out_dir + 'pick_one_mark_nn.pkl')
    joblib.dump(scaler, out_dir + 'pick_one_mark_scaler.pkl')

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
    train_detections.record_scores(prob_train)
    test_detections.record_scores(prob_test)

    # Now test on each view and dump txt for scoring
    for fn in glob.glob(test_dir + '*.feat'):
        folder_name = 'test'
        view_name = fn.split('\\')[-1].split('.feat')[0]

        test_detections_one_view = preprocessing.all_detections_one_view(folder_name, view_name)

        with open(fn) as f:
            lines = f.readlines()
            test_detections_one_view.update(lines, view_name)

        # Get the feature values
        test_x = np.array([test_detections_one_view.get_all_od(), test_detections_one_view.get_all_dl(), test_detections_one_view.get_all_logit()]).transpose()
        test_x = scaler.transform(test_x)
        predictions_test_one_view = mlp.predict_proba(test_x)[:,1]

        # record the predictions
        test_detections_one_view.record_scores(predictions_test_one_view)

        postprocessing.pick_one_mark_one_group_use_prediction(test_detections_one_view)
        postprocessing.save_to_txt(test_detections_one_view, out_dir, view_name)


if __name__ == '__main__':
    run()