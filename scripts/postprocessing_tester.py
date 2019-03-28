import postprocessing, preprocessing
import glob

fn = 'C:/experiments/temp_feature_vector/v1/test/rph101-010-61241-R_CC_feat.txt'


def test_save_to_txt_class(fn):
    out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/'
    folder_name = 'test'
    test_detections = preprocessing.all_detections(folder_name)
    view_name = fn.split('/')[-1].split('_feat.txt')[0]
    with open(fn) as f:
        lines = f.readlines()
    test_detections.update(lines, view_name)
    labels = test_detections.get_all_tp_fp()
    postprocessing.pick_one_mark_one_group(test_detections)
    postprocessing.save_to_txt(test_detections, out_dir)

def run_all_test_max_oddl_save_to_txt(fn):
    out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/'
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        # hard code to make the program run through
        folder_name = 'test'
        test_detections = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
        test_detections.update(lines, view_name)
        labels = test_detections.get_all_tp_fp()
        postprocessing.pick_one_mark_one_group(test_detections)
        postprocessing.save_to_txt(test_detections, out_dir)

def run_all_test_weighted_logit_save_to_txt(fn):
    out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/weighted_logit/'
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        # hard code to make the program run through
        folder_name = 'test'
        test_detections = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
        test_detections.update(lines, view_name)
        labels = test_detections.get_all_tp_fp()
        postprocessing.pick_one_mark_one_group_use_logit(test_detections)
        postprocessing.save_to_txt(test_detections, out_dir)

def test_form_lstm_features(fn):
    out_dir = '//lyta/tomodev/DeepLearning/Playground_stage3/output_pick1mark/score_txt/weighted_logit/'
    for fn in glob.glob('C:/experiments/temp_feature_vector/test/*.txt'):
        # hard code to make the program run through
        folder_name = 'test'
        test_detections = preprocessing.all_detections(folder_name)

        view_name = fn.split('\\')[-1].split('_feat.txt')[0]
        with open(fn) as f:
            lines = f.readlines()
        test_detections.update(lines, view_name)
        lstm_features = postprocessing.form_lstm_features(test_detections)
        print(' ')



def main():
    test_form_lstm_features(fn)

if __name__ == '__main__':
    main()