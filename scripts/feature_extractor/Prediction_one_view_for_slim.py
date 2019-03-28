# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:09:17 2017

@author: lwei
@author: xliu
"""

from os import listdir
from math import ceil
import os
import os.path

import tensorflow as tf
import numpy as np
import time
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from train_lesionNormal import get_split, load_batch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim
#from sklearn.metrics import roc_curve, auc

#import calc_data_retriever as rt
#import visualization as vs

#State your log directory where you can retrieve your model
log_dir = '//lyta/tomodev/DeepLearning/Run01_HighRes_Version01/Stage2-DL/Exp2h-trained_model/'

#Create a new evaluation log directory to visualize the validation process
log_eval = '/log_eval_test'

#State the dataset directory where the validation set is found
#dataset_dir = 'Y:/CAD Science/Projects/database/3DMass/DataSet2/data/TP_Train_Full/'

#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 50

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)

sx = int(256)
sy = int(256) 

idxLabel = int(0)
idxIdx = int(1)
idxCtrX = int(3)
idxCtrY = int(4)
idxCtrZ = int(5)
idxSx = int(6)
idxSy = int(7)
idxProb = int(15)

# #dirPred1 = '//lyta/tomodev/DeepLearning/Playground_stage3/pick_marker/results/DL_features/'
# dirPred1 = 'C:/experiments/temp_feature_vector/'
#
#
# list1 = '//lyta/tomodev/DeepLearning/Lists/highRes/split1/Cancer_Test_Whole_Slice.lis'
# list2 = '//lyta/tomodev/DeepLearning/Lists/highRes/split1/test_normal_set.lis'
#
#
#
#
# dirROI1 = 'Y:/CAD Science/Projects/database/3DMass/DataSet2/data/TP_Test_Full/'
# dirROI2 = 'Y:/CAD Science/Projects/database/3DMass/DataSet2/data/FP_Peaks/'

dirPred = ['C:/experiments/temp_feature_vector/FP_train/','C:/experiments/temp_feature_vector/FP_test/','C:/experiments/temp_feature_vector/TP_train/','C:/experiments/temp_feature_vector/TP_test/']
list_total = ['//lyta/tomodev/DeepLearning/Lists/highRes/split1/train_normal_set.lis','//lyta/tomodev/DeepLearning/Lists/highRes/split1/test_normal_set.lis','//lyta/tomodev/DeepLearning/Lists/highRes/split1/Cancer_Train_Whole_Slice.lis','//lyta/tomodev/DeepLearning/Lists/highRes/split1/Cancer_Test_Whole_Slice.lis']
dirROI = ['Y:/CAD Science/Projects/database/3DMass/DataSet2/data/FP_Peaks/','Y:/CAD Science/Projects/database/3DMass/DataSet2/data/FP_Peaks/','Y:/CAD Science/Projects/database/3DMass/DataSet2/data/TP_Train_Full/','Y:/CAD Science/Projects/database/3DMass/DataSet2/data/TP_Test_Full/']

isNewLis = False
isDetectionData = True

substrInfoFileName = '_fin.info'

img_size = 256
channel = 1
img_size_flat = img_size * img_size

input_size = 299



def preproc(input_images, height=input_size, width=input_size):
    '''
    preprocess the input image for evaluation using inception-resnet-v2

    '''
    images = [];
    raw_images = [];
    for i in range(50):
        raw_image = input_images[i];
        #Perform the correct preprocessing for this image depending if it is training or evaluating
        image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training=False)
        image = tf.expand_dims(image, 0)
        
        #As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.expand_dims(raw_image, 0)
        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
        
        if i == 0:
            images = image
            raw_images = raw_image
        else:
            images = tf.concat([images, image], 0)
            raw_images = tf.concat([raw_images, raw_image], 0)
        
    return images, raw_images

if __name__ == '__main__':
    tf.__version__
    start_time = time.time()

    for i in range(4):
        dirPred1 = dirPred[i]
        list1 = list_total[i]
        dirROI1 = dirROI[i]
    
        if not os.path.exists(log_eval):
            os.mkdir(log_eval)

     #   sess = tf.InteractiveSession()

        img_size = 256
        channel = 1
        img_size_flat = img_size * img_size

        if isDetectionData is True:
            data_bits = 'uint8'
        else:
            data_bits = 'uint16'

        #Just construct the graph from scratch again
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)
            #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
            input_images = tf.placeholder(dtype=tf.uint8, shape=(50, 256, 256, 3))
            input_labels = tf.placeholder(dtype=tf.int32, shape = (50))
            images, raw_images = preproc(input_images)
            labels = input_labels


            #Now create the inference model but set is_training=False
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(images, num_classes = 2, is_training = False)

            # #get all the variables to restore from the checkpoint file and create the saver function to restore
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            def restore_fn(sess):
                return saver.restore(sess, checkpoint_file)

            # Extract the features here
            feature_vector =end_points['PreLogitsFlatten']

            #Just define the metrics to track without the loss or whatsoever
            predictions = tf.argmax(end_points['Predictions'], 1)
            probabilities = end_points['Predictions']
    #        lesion_probability = probabilities[:,0]
    #        lesion_labels = 1 - labels
            lesion_probability = probabilities[:,1]
            lesion_labels = labels
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            auc, auc_update = tf.contrib.metrics.streaming_auc(predictions=lesion_probability, labels=lesion_labels, curve='ROC')
            metrics_op = tf.group(accuracy_update, auc_update)

            #Create the global step and an increment op for monitoring
            global_step = get_or_create_global_step()
            global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step


            #Create a evaluation step function
            def eval_step(sess, metrics_op, global_step):
                '''
                Simply takes in a session, runs the metrics op and some logging information.
                '''
                start_time = time.time()
                #_, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
                _, global_step_count, accuracy_value, auc_value = sess.run([metrics_op, global_step_op, accuracy, auc])
                time_elapsed = time.time() - start_time

                #Log some information
                #logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
                logging.info('Global Step %s: Streaming Accuracy: %.4f Streaming AUC: %.4f (%.2f sec/step)', global_step_count, accuracy_value, auc_value, time_elapsed)

                return accuracy_value


            #Define some scalar quantities to monitor
            tf.summary.scalar('Validation_Accuracy', accuracy)
            tf.summary.scalar('Validation_AUC', auc)
            my_summary_op = tf.summary.merge_all()

            #Get your supervisor
            sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)

            #Now we are ready to run in one session
            with sv.managed_session() as sess:
                sess.run(sv.global_step)

                nNeg = int(0)
                nPos = int(0)

                idxFile = 0

                image_batch = []

                #evaluate lesion list
                with open(list1) as fileLesionLis:
                    LesionLis = fileLesionLis.read().splitlines()
                    for line in LesionLis:
                        if isNewLis is True:
                            line = line.split('\\')[-1].split('.')[0]#for new list only
                        print(('[%3d] %s')%(idxFile, line))
                        idxFile = idxFile + 1
                        allRois     = []
                        allLabels   = []
                        allIdx      = []
                        allCtrX     = []
                        allCtrY     = []
                        allCtrZ     = []
                        allShiftZ   = []
                        allProb     = []
                        allDiffZ     = []

                        a= line.find('rph')
                        b= line.find('.dcm')

                        baseName = line[a:b]

                        # read image data

                        orgName = dirROI1 + baseName + '_ori.info'

                        if (os.path.isfile(orgName)==False):
                            continue;

                        dataName = dirROI1 + baseName + '.data'
                        if (os.path.isfile(dataName)==False):
                            continue;

                        # print(dataName)
                        with open(dataName, 'rb') as fileData:
                            data = np.fromfile(fileData, dtype=data_bits)

                            # get ROI data/label, append to lists
                        infoName = dirROI1 + baseName + '_fin.info'
                        with open(infoName) as fileInfo:
                            curRoiStt = int(0)

                            lines = fileInfo.read().splitlines()
                            for line in lines:
                                tmp = line.split()
                                label = int(tmp[idxLabel])
                                idx = int(tmp[idxIdx])
                                ctrX = int(tmp[idxCtrX])
                                ctrY = int(tmp[idxCtrY])
                                ctrZ = int(tmp[idxCtrZ])
                                    #                sx = int(tmp[idxSx])
                                    #                sy = int(tmp[idxSy])
                                prob = float(tmp[idxProb])
                                    #truZ = int(tmp[idxTru])

                                roi = data[curRoiStt:(curRoiStt+sx*sy)]
                                if label is 0:
                                    nNeg = nNeg + 1
                                else:
                                    label = 1
                                    nPos = nPos + 1

                                allRois.append(roi)
                                allLabels.append(label)
                                allIdx.append(idx)
                                allCtrX.append(ctrX)
                                allCtrY.append(ctrY)
                                allCtrZ.append(ctrZ)
                                allProb.append(prob)
            #                   allDiffZ.append(truZ-ctrZ)

                                curRoiStt = curRoiStt+sx*sy

                        images_all = np.asanyarray(allRois)
                        #images_all = 255 - images_all
                        labels_all = np.asanyarray(allLabels)

                        images_all = images_all.astype(np.uint8)
            #            images_all = images_all.astype(np.float32)
            #        images = np.multiply(images, 1.0 / 4095.0)
            #           images = np.multiply(images, 1.0/255.0)

                        vali_num = len(labels_all)
            #        print('vali_num = %d'%vali_num)

                        timer1 = time.time()


                        training_batch_size = 50
                        last_batch = int(vali_num/training_batch_size)
                    #        print('last_batch=%d'%last_batch)

                        tmpNum = training_batch_size * last_batch
                        if tmpNum == vali_num:
                            residual_num = 0
                        else:
                            residual_num = training_batch_size*(last_batch+1) - vali_num
            #        print('residual_num=%d'%residual_num)


                    #evaluate in a batch fashion
                        cls_pred_cls = []
                        pred_softmax = []
                        logits_lis = []
                        feature_vector_lis = []
                        total_cor = 0

                        if vali_num<50:
                            repeat_rate = int(training_batch_size/vali_num)
                            residule_rep = training_batch_size - int(training_batch_size/vali_num)*vali_num

                            image_batch = images_all[0:vali_num, :]
                            label_batch = labels_all[0:vali_num]
                            if repeat_rate >= 1:
                                for idx_repeat in range(repeat_rate-1):
                                    image_batch = np.concatenate(
                                        (image_batch, images_all[0:vali_num, :]),
                                        axis=0)
                                    label_batch = np.concatenate(
                                        (label_batch, labels_all[0:vali_num]))
                            image_batch = np.concatenate((image_batch, images_all[0:residule_rep, :]),
                                                         axis=0).reshape(50, sx, sy, 1)
                            label_batch = np.concatenate((label_batch, labels_all[0:residule_rep]))

                            # if repeat_rate == 2:
                            #     image_batch = np.concatenate((images_all[0:vali_num,:], images_all[0:vali_num,:], images_all[0:residule_rep,:]), axis = 0).reshape(50, sx, sy, 1)
                            #     label_batch = np.concatenate((labels_all[0:vali_num], labels_all[0:vali_num], labels_all[0:residule_rep]))
                            #         #image_batch = np.concatenate((image_batch[0:42,:],images[0,residule_rep,:]), axis = 0)
                            #         #label_batch = np.concatenate((label_batch,labels[0:residule_rep]))
                            # if repeat_rate == 1:
                            #     image_batch = np.concatenate((images_all[0:vali_num,:], images_all[0:residule_rep,:]), axis = 0).reshape(50, sx, sy, 1)
                            #     label_batch = np.concatenate((labels_all[0:vali_num:],labels_all[0:residule_rep]))

                            rgb_image_batch = np.repeat(image_batch, 3, 3)
                            raw_imgs, imgs, lbls, preds, probs, _, global_step_count, accuracy_value, auc_value, logits_value, feature_vector_value = sess.run([raw_images, images, labels, predictions, lesion_probability, metrics_op, global_step_op, accuracy, auc, logits, feature_vector], feed_dict={input_images:rgb_image_batch, input_labels:label_batch})
                            #Log some information
                            logging.info('Global Step %s: Streaming Accuracy: %.4f Streaming AUC: %.4f ', global_step_count, accuracy_value, auc_value)

                            cls_pred_cls.extend(preds[0:vali_num])
                            pred_softmax.extend(probs[0:vali_num])
                            feature_vector_lis.extend(feature_vector_value[0:vali_num])
                            logits_lis.extend(logits_value[0:vali_num])
    #                        cls_pred_cls_batch, correct, pred_softmax_batch = sess.run([y_pred_cls, correct_prediction, y_pred_softmax], feed_dict={x: image_batch, y_true_cls: label_batch, keep_prob: 1.0})
    #                        cls_pred_cls.extend(cls_pred_cls_batch[0:vali_num])
    #                        pred_softmax.extend(pred_softmax_batch[0:vali_num:,1])
    #                        total_cor += correct[0:vali_num].sum()

                        if vali_num>=50:
                            for i in range(int(vali_num/training_batch_size)):
                                image_batch = images_all[i*50:i*50+50,:].reshape(50, sx, sy, 1)
                                rgb_image_batch = np.repeat(image_batch, 3, 3)
            #                    for j in range(50):
            #                        plt.imshow(rgb_image_batch[j])
            #                        plt.show()
                                label_batch = labels_all[i*50:i*50+50]
                                raw_imgs, imgs, lbls, preds, probs, _, global_step_count, accuracy_value, auc_value, logits_value, feature_vector_value = sess.run([raw_images, images, labels, predictions, lesion_probability, metrics_op, global_step_op, accuracy, auc, logits, feature_vector], feed_dict={input_images:rgb_image_batch, input_labels:label_batch})
                                #Log some information
                                logging.info('Global Step %s: Streaming Accuracy: %.4f Streaming AUC: %.4f ', global_step_count, accuracy_value, auc_value)

                                #for i in range(50):
                                    #img, lbl, pred, prob = imgs[i], lbls[i], preds[i], probs[i]
                                    #img, lbl, pred, prob = raw_imgs[i], lbls[i], preds[i], probs[i]
                                    #if (lbl != pred) and (lbl == 0):
    #                                if lbl == 1:
    #                                    prediction_name, label_name = dataset.labels_to_name[pred], dataset.labels_to_name[lbl]
    #                                    text = 'Prediction: %s    Ground Truth: %s \n Lesion Probability: %.4f' %(prediction_name, label_name, prob)
    #                                    img_plot = plt.imshow(img)
    #                                    #Set up the plot and hide axes
    #                                    plt.title(text)
    #                                    img_plot.axes.get_yaxis().set_ticks([])
    #                                    img_plot.axes.get_xaxis().set_ticks([])
    #                                    plt.show()


                                cls_pred_cls.extend(preds)
                                pred_softmax.extend(probs)
                                feature_vector_lis.extend(feature_vector_value)
                                logits_lis.extend(logits_value)

                              #        print('Batch %d' % (i))

                                #evaluate the last batch
                            if residual_num > 0:
                                image_batch = np.concatenate((images_all[last_batch*50:,:], images_all[0:residual_num,:]), axis = 0).reshape(50, sx, sy, 1)
                                label_batch = np.concatenate((labels_all[last_batch*50:], labels_all[0:residual_num]))
                                raw_imgs, lbls, preds, probs, _, global_step_count, accuracy_value, auc_value, logits_value, feature_vector_value = sess.run([raw_images, labels, predictions, lesion_probability, metrics_op, global_step_op, accuracy, auc, logits, feature_vector], feed_dict={input_images:rgb_image_batch, input_labels:label_batch})
                                #Log some information
                                logging.info('Global Step %s: Streaming Accuracy: %.4f Streaming AUC: %.4f ', global_step_count, accuracy_value, auc_value)

                                cls_pred_cls.extend(preds[0:-residual_num])
                                pred_softmax.extend(probs[0:-residual_num])
                                feature_vector_lis.extend(feature_vector_value[0:-residual_num])
                                logits_lis.extend(logits_value[0:-residual_num])

            #            total_cor = total_cor/vali_num
            #            #        print('test acc %.4f' % (total_cor))
            #
                        cls_pred_cls = np.asanyarray(cls_pred_cls)
                        feature_vector_lis = np.asanyarray(feature_vector_lis)
                        logits_lis = np.asanyarray(logits_lis)
                        cls_true_cls = labels_all
                        #    vs.plot_confusion_matrix(cls_true_cls, cls_pred_cls, 2)

                            #plot ROC curve
                        pred_softmax = np.asanyarray(pred_softmax)
                        #        print(pred_softmax)
                        #        print(len(pred_softmax))
                        #        print(len(cls_true_cls))
                        nSample = len(cls_true_cls)
                        aug = 60
                        #aug = 20

                        filenamePred = dirPred1 + baseName  + '_score.txt'

                        with open(filenamePred, 'wt') as fp:
                            for i in range(nSample):
                                output = []
                                output.extend([cls_true_cls[i]])
                                output.extend([allIdx[i]])
                                output.extend([allCtrX[i]])
                                output.extend([allCtrY[i]])
                                output.extend([allCtrZ[i]])
                                output.extend([allProb[i]])
                                output.extend([pred_softmax[i]])
                                output.extend([logits_lis[i][0]])# comment out this line if only need one logit
                                output.extend([logits_lis[i][1]])
                                output.extend(feature_vector_lis[i])
                                for item in output:
                                    fp.write("%s\t" % item)
                                fp.write('\n')
                                # fp.write('%d %d %4d %4d %3d %.3f %.3f\n'%(cls_true_cls[i], allIdx[i], allCtrX[i], allCtrY[i], allCtrZ[i], allProb[i], pred_softmax[i]))

                        # filenamePred = dirPred2 + baseName  + '_score.txt'
                        # with open(filenamePred, 'wt') as fp:
                        #     for i in range(nSample):
                        #         #if(i== int(i/aug)*aug+20):
                        #         if(i== int(i/aug)*aug+4):
                        #             fp.write('%d %4d %4d %3d %.3f %.3f\n'%(cls_true_cls[i], allCtrX[i], allCtrY[i], allCtrZ[i], allProb[i], pred_softmax[i]))
                        #
                        # filenamePred = dirPred3 + baseName  + '_score.txt'
                        # temp_sum=0.0
                        # with open(filenamePred, 'wt') as fp:
                        #     for i in range(nSample):
                        #         if i >= int(i/aug)*aug and i < int(i/aug+1)*aug:
                        #             temp_sum = temp_sum + pred_softmax[i]
                        #             #if temp_sum<pred_softmax[i]:
                        #             #    temp_sum = pred_softmax[i]
                        #
                        #         if (i==int(i/aug+1)*aug-1):
                        #             fp.write('%d %4d %4d %3d %.3f %.3f\n'%(cls_true_cls[i], allCtrX[i], allCtrY[i], allCtrZ[i], allProb[i], temp_sum/aug))
                        #             #fp.write('%d %4d %4d %3d %.3f %.3f\n'%(cls_true_cls[i], allCtrX[i], allCtrY[i], allCtrZ[i], allProb[i], temp_sum))
                        #             temp_sum=0.0
                        #
                        # filenamePred = dirPred4 + baseName  + '_score.txt'
                        # temp_sum=0.0
                        # with open(filenamePred, 'wt') as fp:
                        #     for i in range(nSample):
                        #         if i >= int(i/aug)*aug and i < int(i/aug+1)*aug:
                        #             if temp_sum<pred_softmax[i]:
                        #                 temp_sum = pred_softmax[i]
                        #
                        #         if (i==int(i/aug+1)*aug-1):
                        #             fp.write('%d %4d %4d %3d %.3f %.3f\n'%(cls_true_cls[i], allCtrX[i], allCtrY[i], allCtrZ[i], allProb[i], temp_sum))
                        #             temp_sum=0.0



        print("Total run time: %.2f sec"%(time.time()-start_time))
        

