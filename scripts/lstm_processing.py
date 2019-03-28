import numpy as np
import os


def form_lstm_features(detections):
    total_group_count = detections.get_total_group_count()
    dl_feat_1536_all = []
    for i in range(total_group_count):
        dl_feat_1536_group = []
        for j in range(detections.groups[i].detection_count):
            dl_feat_1536_group.append(detections.groups[i].detections[j].dl_feat_1536)
        dl_feat_1536_all.append(dl_feat_1536_group)
    return dl_feat_1536_all


def form_lstm_features_v2(detections):
    # this function use od, dl and logits as lstm features
    total_group_count = detections.get_total_group_count()
    dl_feat_3_all = []
    for i in range(total_group_count):
        dl_feat_3_group = []
        for j in range(detections.groups[i].detection_count):
            dl_feat_3_group.append([detections.groups[i].detections[j].od, detections.groups[i].detections[j].dl, detections.groups[i].detections[j].logit])
        dl_feat_3_all.append(dl_feat_3_group)
    return dl_feat_3_all


def form_lstm_features_v3(detections):
    # this function use od, dl logits and z as lstm features
    total_group_count = detections.get_total_group_count()
    dl_feat_4_all = []
    for i in range(total_group_count):
        dl_feat_4_group = []
        for j in range(detections.groups[i].detection_count):
            dl_feat_4_group.append([detections.groups[i].detections[j].od, detections.groups[i].detections[j].dl, detections.groups[i].detections[j].logit, detections.groups[i].detections[j].z])
        dl_feat_4_all.append(dl_feat_4_group)
    return dl_feat_4_all


def repeat_and_pad(features, maxlen):
    # instead of padding zeros, we repeat the sequence and duplicate to certain length.
    features_padded = []
    for i in range(len(features)):
        if len(features[i]) >= maxlen:
            features_padded.append(features[i][0:maxlen])
        else:
            n = np.floor(maxlen/len(features[i]))
            feat_rep = features[i] * int(n+1)
            assert len(feat_rep)>=maxlen
            features_padded.append(feat_rep[0:maxlen])
    return features_padded


def insert_zero_form_sequence(features):
    # for non-detection slices, insert 0s, so the detections match the actual z locations
    feature_new = []
    for i in range(len(features)):
        z_max = max([z[-1] for z in features[i]])
        feature_new_one_detection = np.zeros([z_max+1, len(features[i][0])-1])
        for j in range(len(features[i])):
            z = features[i][j][-1]
            feature_new_one_detection[z][:] = features[i][j][0:len(features[i][0]) - 1]
        feature_new.append(feature_new_one_detection)
    return feature_new


def save_to_txt_no_pick_one_mark(all_detections, out_dir):
    # dump txt files for group based scoring using lstm features
    total_group_count = all_detections.get_total_group_count()

    with open(os.path.join(out_dir, all_detections.groups[0].view_name + '_score.txt'), 'w') as f:
        for i in range(total_group_count):
            for j in range(all_detections.groups[i].detection_count):
                output = []
                output.extend([all_detections.groups[i].detections[j].tp_fp])
                output.extend([all_detections.groups[i].detections[j].x])
                output.extend([all_detections.groups[i].detections[j].y])
                output.extend([all_detections.groups[i].detections[j].z])
                output.extend([all_detections.groups[i].detections[j].od])
                output.extend([all_detections.groups[i].group_lstm_score])
                for item in output:
                    f.write("%s\t" % item)
                f.write('\n')