
import preprocessing

fn = '//lyta/tomodev/DeepLearning/Run11_PrototypeV3_streamline/LowRes2/Output_exp2h_OD_V2_DL_4x_scored//rp003-999-00075-R_CC_exp2h_score.feat'
def test_detection_class(fn):
    view_name = fn.split('/')[-1].split('.feat')[0]
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            one_detection = preprocessing.detection(line, view_name)
            print(one_detection.view_name)

def test_group_class(fn):
    view_name = fn.split('/')[-1].split('.feat')[0]
    with open(fn) as f:
        lines = f.readlines()
        targets = []
        for line in lines:
            one_detection = preprocessing.detection(line, view_name)
            if one_detection.groupid == 4:
                targets.append(one_detection)
        one_group = preprocessing.group(targets, view_name)
        print(one_group.detections[0].groupid)

def test_all_detections_class(fn):
    folder_name = 'test'
    test_detections = preprocessing.all_detections(folder_name)
    view_name = fn.split('/')[-1].split('.feat')[0]
    with open(fn) as f:
        lines = f.readlines()
    test_detections.update(lines, view_name)
    labels = test_detections.get_all_tp_fp()
    print(' ')

def main():
    test_group_class(fn)

if __name__ == '__main__':
    main()