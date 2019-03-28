
import os
import numpy as np

def calculate_weighted_average(z_idx, scores):
    # shift all values to non-zeros
    if len(scores) > 1 and (sum(np.array(scores) - min(scores)) != 0):
        scores = np.array(scores) - min(scores)

    return int(round(sum(np.multiply(np.array(z_idx), np.array(scores))) / sum(scores)))

def find_nearest_idx(vector,value):
    idx = (np.abs(np.array(vector)-value)).argmin()
    return idx


def pick_one_mark_one_group_use_prediction(all_detections):
    # Use max prediction to pick a slice, will update the detection instance
    total_group_count = all_detections.get_total_group_count()

    for i in range(total_group_count):

        for j in range(all_detections.groups[i].detection_count):
            all_detections.groups[i].detections[j].ispick = 0 # set all ispick to 0 at beginning

        all_detections.groups[i].sort_detection_by_z()

        prediction_vector_per_group = [all_detections.groups[i].detections[j].prediction for j in range(all_detections.groups[i].detection_count)]
        all_detections.groups[i].picked_slice_reference_score = max(prediction_vector_per_group)


        # method 1: pick the center

        # max_score_slice_idx = [idx if all_detections.groups[i].picked_slice_reference_score == x else None for idx, x in enumerate(prediction_vector_per_group)]
        # max_score_slice_idx = list(filter(lambda x: x is not None, max_score_slice_idx))
        # all_detections.groups[i].picked_slice_idx = int((max(max_score_slice_idx) + min(max_score_slice_idx))/2)

        #method 2: gravity

        # zs = [all_detections.groups[i].detections[j].z for j in range(all_detections.groups[i].detection_count)]
        # z_selected = calculate_weighted_average(zs, prediction_vector_per_group)
        # all_detections.groups[i].picked_slice_idx = find_nearest_idx(zs,z_selected)
        # all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].ispick = 1
        # all_detections.groups[i].picked_slice_reference_score = all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].prediction

        # method 3: center z, same as Jin-Long's method
        max_score_slice_idx = [idx if all_detections.groups[i].picked_slice_reference_score == x else None for idx, x in enumerate(prediction_vector_per_group)]
        max_score_slice_idx = list(filter(lambda x: x is not None, max_score_slice_idx))
        if len(max_score_slice_idx) == 1:
            all_detections.groups[i].picked_slice_idx = int(max_score_slice_idx[0])
        else:
            # calculate the center z
            centerZ = all_detections.groups[i].detections[int(all_detections.groups[i].detection_count/2)].z
            distZ = [abs(all_detections.groups[i].detections[iter].z - centerZ) for iter in max_score_slice_idx]
            loc_k = distZ.index(min(distZ))
            all_detections.groups[i].picked_slice_idx = max_score_slice_idx[loc_k]
        all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].ispick = 1

def pick_one_mark_one_group_use_logit(all_detections):
    # Use logits to pick a slice, will update the detection instance
    total_group_count = all_detections.get_total_group_count()

    for i in range(total_group_count):

        for j in range(all_detections.groups[i].detection_count):
            all_detections.groups[i].detections[j].ispick = 0 # set all ispick to 0 at beginning

        all_detections.groups[i].sort_detection_by_z()

        logit_vector_per_group = [all_detections.groups[i].detections[j].logit for j in range(all_detections.groups[i].detection_count)]
        all_detections.groups[i].picked_slice_reference_score = max(logit_vector_per_group)


        # method 1: pick the center

        # max_score_slice_idx = [idx if all_detections.groups[i].picked_slice_reference_score == x else None for idx, x in enumerate(logit_vector_per_group)]
        # max_score_slice_idx = list(filter(lambda x: x is not None, max_score_slice_idx))
        # all_detections.groups[i].picked_slice_idx = int((max(max_score_slice_idx) + min(max_score_slice_idx))/2)

        #method 2: gravity

        zs = [all_detections.groups[i].detections[j].z for j in range(all_detections.groups[i].detection_count)]
        z_selected = calculate_weighted_average(zs, logit_vector_per_group)
        all_detections.groups[i].picked_slice_idx = find_nearest_idx(zs,z_selected)

        all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].ispick = 1
        all_detections.groups[i].picked_slice_reference_score = all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].logit

def pick_one_mark_one_group(all_detections):
    # Use max (od+dl) to pick a slice, will update the detection instance
    total_group_count = all_detections.get_total_group_count()
    for i in range(total_group_count):

        all_detections.groups[i].sort_detection_by_z()

        od_dl_vector_per_group = []
        for j in range(all_detections.groups[i].detection_count):
            all_detections.groups[i].detections[j].ispick = 0 # set all ispick to 0 at beginning
            od_dl_vector_per_group.append(all_detections.groups[i].detections[j].od + all_detections.groups[i].detections[j].dl)
        all_detections.groups[i].picked_slice_reference_score = max(od_dl_vector_per_group)

        max_score_slice_idx = [idx if all_detections.groups[i].picked_slice_reference_score == x else None for idx, x in enumerate(od_dl_vector_per_group)]
        max_score_slice_idx = list(filter(lambda x: x is not None, max_score_slice_idx))

        # method 1: pick the center
        # all_detections.groups[i].picked_slice_idx = int((max(max_score_slice_idx) + min(max_score_slice_idx)) / 2)
        if len(max_score_slice_idx) == 1:
            all_detections.groups[i].picked_slice_idx = int(max_score_slice_idx[0])
        else:
            # calculate the center z
            centerZ = all_detections.groups[i].detections[int(all_detections.groups[i].detection_count/2)].z
            distZ = [abs(all_detections.groups[i].detections[iter].z - centerZ) for iter in max_score_slice_idx]
            loc_k = distZ.index(min(distZ))
            all_detections.groups[i].picked_slice_idx = max_score_slice_idx[loc_k]

        # method 2: pick the weighted average of all the maxes
        scores_max_score_slice = [all_detections.groups[i].detections[j].dl + all_detections.groups[i].detections[j].od for j in max_score_slice_idx]
        #all_detections.groups[i].picked_slice_idx = calculate_weighted_average(max_score_slice_idx, scores_max_score_slice)

        all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx].ispick = 1

def save_to_txt(all_detections, out_dir, view_name):
    # dump score txt files for scoring
    total_group_count = all_detections.get_total_group_count()
    picked_slice = []
    for i in range(total_group_count):
        picked_slice.append(all_detections.groups[i].detections[all_detections.groups[i].picked_slice_idx])

    with open(os.path.join(out_dir, view_name + '_score.txt'), 'w') as f:
        if total_group_count > 0:
            for i in range(total_group_count):
                output = []
                output.extend([picked_slice[i].tp_fp])
                output.extend([picked_slice[i].x])
                output.extend([picked_slice[i].y])
                output.extend([picked_slice[i].z])
                output.extend([picked_slice[i].od])
                output.extend([picked_slice[i].accu_dl])
                f.write("%2d %4d %4d %4d %8.4f %8.4f\n" % (
                    output[0], output[1], output[2], output[3], output[4], output[5]))
                # for item in output:
                #     f.write("%s\t" % item)
                # f.write('\n')
        else:
            pass




# def save_to_feat_txt(all_detections, out_dir):
#     # dump feat txt files, not for scoring
#     total_group_count = all_detections.get_total_group_count()
#
#     with open(os.path.join(out_dir, all_detections.groups[0].view_name + '_feat.txt'), 'w') as f:
#         for i in range(total_group_count):
#             for j in range(all_detections.groups[i].detection_count):
#                 output = []
#                 output.extend([all_detections.groups[i].detections[j].tp_fp])
#                 output.extend([all_detections.groups[i].detections[j].x])
#                 output.extend([all_detections.groups[i].detections[j].y])
#                 output.extend([all_detections.groups[i].detections[j].z])
#                 output.extend([all_detections.groups[i].detections[j].od])
#                 output.extend([all_detections.groups[i].detections[j].dl])
#                 #output.extend([all_detections.groups[i].detections[j].logit_normal])
#                 #output.extend([all_detections.groups[i].detections[j].logit])
#                 output.extend([all_detections.groups[i].detections[j].groupid])
#                 output.extend([all_detections.groups[i].detections[j].ispick])
#                 output.extend([all_detections.groups[i].detections[j].accu_dl])
#                 f.write("%2d %4d %4d %4d %8.4f %8.4f %4d   %1d %8.4f\n" % (output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]))
#                 # for item in output:
#                 #     f.write("%s\t" % item)
#                 # f.write('\n')

def save_to_feat_txt(all_detections, out_dir, view_name):
    # dump feat txt files, not for scoring
    total_group_count = all_detections.get_total_group_count()

    with open(os.path.join(out_dir, view_name + '_grp1mark.feat'), 'w') as f:
        if total_group_count > 0:
            for i in range(total_group_count):
                for j in range(all_detections.groups[i].detection_count):
                    output = []
                    output.extend([all_detections.groups[i].detections[j].tp_fp])
                    output.extend([all_detections.groups[i].detections[j].det_id])
                    output.extend([all_detections.groups[i].detections[j].groupid])
                    output.extend([all_detections.groups[i].detections[j].ispick])
                    output.extend([all_detections.groups[i].detections[j].x])
                    output.extend([all_detections.groups[i].detections[j].y])
                    output.extend([all_detections.groups[i].detections[j].z])
                    output.extend([all_detections.groups[i].detections[j].xmin])
                    output.extend([all_detections.groups[i].detections[j].xmax])
                    output.extend([all_detections.groups[i].detections[j].ymin])
                    output.extend([all_detections.groups[i].detections[j].ymax])
                    output.extend([all_detections.groups[i].detections[j].zmin])
                    output.extend([all_detections.groups[i].detections[j].zmax])
                    output.extend([all_detections.groups[i].detections[j].od])
                    output.extend([all_detections.groups[i].detections[j].dl])
                    output.extend([all_detections.groups[i].detections[j].logit_normal])
                    output.extend([all_detections.groups[i].detections[j].logit])
                    output.extend([all_detections.groups[i].detections[j].accu_dl])
                    f.write("%d %3d %3d %d %4d %4d %4d %9.4f %9.4f %9.4f %9.4f %4d %4d %5.4f %5.4f %8.4f %8.4f %8.4f\n" % (
                    output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8],
                    output[9], output[10], output[11], output[12], output[13], output[14], output[15], output[16], output[17]))
                    # for item in output:
                    #     f.write("%s\t" % item)
                    # f.write('\n')
        else:
            pass