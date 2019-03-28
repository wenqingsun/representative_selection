# this is v2, use 3 features as input, also changed to one lstm layer

from __future__ import print_function

import glob, os
import preprocessing, postprocessing, lstm_processing
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc


from sklearn.preprocessing import StandardScaler

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

# train_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_train_group/'
# test_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/ODV1-Exp2h_test_group/'
train_dir = 'C:/experiments/temp_feature_vector/train/'
test_dir = 'C:/experiments/temp_feature_vector/test/'

out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_groupProc/lstm_based_model/score_txt/v2/'

maxlen = 40
batch_size = 32

def run():
    # get training set
    os.chdir(train_dir)
    folder_name = train_dir.split('/')[-2]
    # initiate train detections
    train_detections = preprocessing.all_detections(folder_name)
    for file in glob.glob('*.txt'):
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
    for file in glob.glob('*.txt'):
        with open(file) as f:
            lines = f.readlines()
            view_name = file.split('_feat.txt')[0]
            # adding detections from one view to this instance
            test_detections.update(lines, view_name)

    # get train_x and train_y
    train_y = np.array([train_detections.groups[i].group_tp_fp for i in range(train_detections.get_total_group_count())])
    train_x = lstm_processing.form_lstm_features_v2(train_detections)
    train_x = sequence.pad_sequences(train_x, maxlen=maxlen, dtype='float')
    train_x = np.array(train_x)

    del train_detections

    # get test_x and test_y
    test_y = np.array([test_detections.groups[i].group_tp_fp for i in range(test_detections.get_total_group_count())])
    test_x = lstm_processing.form_lstm_features_v2(test_detections)
    test_x = sequence.pad_sequences(test_x, maxlen=maxlen, dtype='float')
    test_x = np.array(test_x)

    del test_detections

    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(maxlen, 3)))
    model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(512, return_sequences=True)))
    # model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(128)))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=20,
              validation_data=[test_x, test_y])

    # predictions on train set
    predictions_train = model.predict(train_x)
    loss_train = model.evaluate(train_x, train_y, verbose=1)
    print('This is the training set result: \n')

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(train_y, predictions_train, pos_label=1)
    az_train = auc(fpr, tpr)
    print('AUC for train set is ' + str(az_train))

    # predictions on test set
    predictions_test = model.predict(test_x)
    loss_test = model.evaluate(test_x, test_y, verbose=1)
    print('This is the training set result: \n')

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(test_y, predictions_test, pos_label=1)
    az_test = auc(fpr, tpr)
    print('AUC for testing set is ' + str(az_test))

    # # record the predictions into the class
    # train_detections.record_group_lstm_scores(predictions_train[:,0])
    # test_detections.record_group_lstm_scores(predictions_test[:,0])

    # save model
    model.save_weights(out_dir + 'model.h5')

    # Now test on each view and dump txt for scoring
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        folder_name = 'test'
        test_detections_one_view = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
            test_detections_one_view.update(lines, view_name)

        # Get the feature values
        test_x = lstm_processing.form_lstm_features_v2(test_detections_one_view)
        test_x = sequence.pad_sequences(test_x, maxlen=maxlen, dtype='float')
        test_x = np.array(test_x)

        # predict
        predictions_test_one_view = model.predict(test_x)

        # record the predictions
        test_detections_one_view.record_group_lstm_scores(predictions_test_one_view[:,0])
        lstm_processing.save_to_txt_no_pick_one_mark(test_detections_one_view, out_dir)



if __name__ == '__main__':
    run()