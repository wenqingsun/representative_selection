'''
By Wenqing May 10, 2018
use logits, then get the weighted average, pick the closest slice.
'''

import postprocessing, preprocessing
import glob

def run_all_test_weighted_logit_save_to_txt():
    out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/weighted_logit/'
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        # hard code to make the program run through
        folder_name = 'test'
        test_detections = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
        test_detections.update(lines, view_name)
        postprocessing.pick_one_mark_one_group_use_logit(test_detections)
        postprocessing.save_to_txt(test_detections, out_dir)

def main():
    run_all_test_weighted_logit_save_to_txt()

if __name__ == '__main__':
    main()