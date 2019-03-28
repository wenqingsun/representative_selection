import glob, os


class detection_old:
    # one instance is one detection
    # used for odv1, before may 16, 2018
    def __init__(self, line, view_name):
        self._attributes = line.split()
        self._view_name = view_name

    @property
    def tp_fp(self):
        return int(self._attributes[0])

    @property
    def x(self):
        return int(self._attributes[1])

    @property
    def y(self):
        return int(self._attributes[2])

    @property
    def z(self):
        return int(self._attributes[3])

    @property
    def od(self):
        return float(self._attributes[4])

    @property
    def dl(self):
        return float(self._attributes[5])

    @property
    def groupid(self):
        return int(self._attributes[6])

    @property
    def ispick(self):
        return int(self._attributes[7])

    @property
    def accu_dl(self):
        return float(self._attributes[8])

    @property
    def logit(self):
        return float(self._attributes[10])

    # @property
    # def dl_feat_1536(self):
    #     return [float(x) for x in self._attributes[11:]]

    @property
    def view_name(self):
        return self._view_name

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self,prediction):
        self._prediction = float(prediction)



# class detection_streamline_v1:
#     # one instance is one detection
#     # used for streamline, since may 16, 2018
#     def __init__(self, line, view_name):
#         attributes = line.split()
#         self.tp_fp = int(attributes[0])
#         self.x = int(attributes[1])
#         self.y = int(attributes[2])
#         self.z = int(attributes[3])
#         self.od = float(attributes[4])
#         self.dl = float(attributes[5])
#         self.groupid = int(attributes[6])
#         self.ispick = int(attributes[7])
#         self.accu_dl = float(attributes[8])
#         # self.logit_normal = float(attributes[6])
#         # self.logit = float(attributes[7])
#         #self.dl_feat_1536 = [float(x) for x in attributes[11:]]
#         self.view_name = view_name
#         self.prediction = 0
#
#     def update_prediction(self, prediction):
#         self.prediction = prediction

class detection_v2:
    # one instance is one detection
    # used for streamline, since may 16, 2018
    def __init__(self, line, view_name):
        self._attributes = line.split()
        self._view_name = view_name
        self.ispick = int(self._attributes[3])


    @property
    def tp_fp(self):
        return int(self._attributes[0])

    @property
    def det_id(self):
        return int(self._attributes[1])

    @property
    def groupid(self):
        return int(self._attributes[2])

    @property
    def ispick(self):
        return self._ispick

    @ispick.setter
    def ispick(self,pick):
        self._ispick = int(pick)

    @property
    def x(self):
        return int(self._attributes[4])

    @property
    def y(self):
        return int(self._attributes[5])

    @property
    def z(self):
        return int(self._attributes[6])

    @property
    def xmin(self):
        return float(self._attributes[7])

    @property
    def xmax(self):
        return float(self._attributes[8])

    @property
    def ymin(self):
        return float(self._attributes[9])

    @property
    def ymax(self):
        return float(self._attributes[10])

    @property
    def zmin(self):
        return float(self._attributes[11])

    @property
    def zmax(self):
        return float(self._attributes[12])

    @property
    def od(self):
        return float(self._attributes[13])

    @property
    def dl(self):
        return float(self._attributes[14])

    @property
    def logit_normal(self):
        return float(self._attributes[15])

    @property
    def logit(self):
        return float(self._attributes[16])

    @property
    def accu_dl(self):
        return float(self._attributes[17])

    # @property
    # def dl_feat_1536(self):
    #     return [float(x) for x in self._attributes[11:]]

    @property
    def view_name(self):
        return self._view_name

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self,one_prediction):
        self._prediction = float(one_prediction)

class detection(detection_v2):
    # fit both 18 and 28 digits
    def __init__(self, line, view_name):
        self._attributes = line.split()
        if len(self._attributes) ==18:
            super().__init__(line, view_name)
        elif len(self._attributes) == 28:
            super().__init__(line, view_name)
            self.logit_normal1 = float(self._attributes[17])
            self.logit1 = float(self._attributes[18])
            self.logit_normal2 = float(self._attributes[19])
            self.logit2 = float(self._attributes[20])
            self.logit_normal3 = float(self._attributes[21])
            self.logit3 = float(self._attributes[22])
            self.logit_normal4 = float(self._attributes[23])
            self.logit4 = float(self._attributes[24])
            self.logit_normal5 = float(self._attributes[25])
            self.logit5 = float(self._attributes[26])

            @property
            def accu_dl(self):
                return float(self._attributes[27])


class group:
    # one instance is one group
    def __init__(self, iterable, view_name):
        self._view_name = view_name
        self._iterable = iterable

    @property
    def detections(self):
        groupid = []
        tp_fp = []
        self._detections = []
        for line in self._iterable:
            self._detections.append(line)
            groupid.append(line.groupid)
            tp_fp.append(line.tp_fp)
        assert len(set(groupid)) == 1
        self.groupid = groupid[0]
        self.group_tp_fp = max(tp_fp)
        return self._detections

    @property
    def group_dl_feature_vector(self):
        group_dl_feature_vector = []
        for detection in self.detections:
            group_dl_feature_vector.append(detection.dl_feat_1536)
        assert len(group_dl_feature_vector) == self.detection_count
        return group_dl_feature_vector

    @property
    def view_name(self):
        return self._view_name

    # @property
    # def groupid(self):
    #     return None
    #
    # @property
    # def group_tp_fp(self):
    #     return None

    @property
    def group_lstm_score(self):
        return self._group_lstm_score

    @property
    def detection_count(self):
        return len(self._iterable)

    @property
    def picked_slice_idx(self):
        return self._picked_slice_idx

    @picked_slice_idx.setter
    def picked_slice_idx(self, value):
        self._picked_slice_idx = value

    @property
    def picked_slice_reference_score(self):
        return self._picked_slice_reference_score

    @picked_slice_reference_score.setter
    def picked_slice_reference_score(self, value):
        self._picked_slice_reference_score = value


    @group_dl_feature_vector.setter
    def group_dl_feature_vector(self):
        for detection in self.detections:
            self.group_dl_feature_vector.append(detection.dl_feat_1536)
        assert len(self.group_dl_feature_vector) == self.detection_count


    def sort_detection_by_z(self):
        self.detections.sort(key=lambda x:x.z)

class all_detections:
    # one instance is either train or test
    # Need to initiate first, then use update func to add more txt data in
    def __init__(self, folder_name):
        self.groups = []
        self.folder_name = folder_name


    def update(self, iterable, view_name, update_tp_group_only = False):
        detections = []
        all_groupid = []
        for idx, line in enumerate(iterable):
            this_detection = detection(line, view_name)
            detections.append(this_detection)
            all_groupid.append(this_detection.groupid)
        unique_groupid = set(all_groupid)
        for i in unique_groupid:
            one_group_detection = []
            for j in detections:
                if j.groupid == i:
                    one_group_detection.append(j)
            one_group = group(one_group_detection, view_name)
            one_group.sort_detection_by_z()
            if update_tp_group_only == False:
                self.groups.append(one_group)
            elif update_tp_group_only==True:
                if one_group.group_tp_fp == 1:
                    self.groups.append(one_group)


    def get_total_group_count(self):
        return len(self.groups)

    def get_total_detection_count(self):
        total_detections = 0
        for i in self.groups:
            total_detections += i.detection_count
        return total_detections

    def get_all_tp_fp(self):
        total_group_count = self.get_total_group_count()
        labels = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                labels.append(self.groups[i].detections[j].tp_fp)
        return labels

    def get_all_od(self):
        total_group_count = self.get_total_group_count()
        od = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                od.append(self.groups[i].detections[j].od)
        return od

    def get_all_dl(self):
        total_group_count = self.get_total_group_count()
        dl = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                dl.append(self.groups[i].detections[j].dl)
        return dl

    def get_all_logit(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit)
        return logit

    def get_all_logit1(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit1)
        return logit

    def get_all_logit2(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit2)
        return logit

    def get_all_logit3(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit3)
        return logit

    def get_all_logit4(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit4)
        return logit

    def get_all_logit5(self):
        total_group_count = self.get_total_group_count()
        logit = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                logit.append(self.groups[i].detections[j].logit5)
        return logit

    def get_all_dl_feat_1536(self):
        total_group_count = self.get_total_group_count()
        dl_feat_1536 = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                dl_feat_1536.append(self.groups[i].detections[j].dl_feat_1536)
        return dl_feat_1536

    def get_all_area(self):
        total_group_count = self.get_total_group_count()
        det_area = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                x_dis = self.groups[i].detections[j].xmax - self.groups[i].detections[j].xmin
                y_dis = self.groups[i].detections[j].ymax - self.groups[i].detections[j].ymin
                det_area.append(abs(x_dis) * abs(y_dis))
        return det_area

    def get_all_detection_count(self):
        total_group_count = self.get_total_group_count()
        det_count = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                det_count.append(self.groups[i].detection_count)
        return det_count

    def get_all_z(self):
        total_group_count = self.get_total_group_count()
        zs = []
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                zs.append(self.groups[i].detections[j].z)
        return zs

    def record_scores(self,predictions):
        total_detection_count = self.get_total_detection_count()
        total_group_count = self.get_total_group_count()
        assert len(predictions) == total_detection_count
        count = 0
        for i in range(total_group_count):
            for j in range(self.groups[i].detection_count):
                self.groups[i].detections[j].prediction = predictions[count]
                count += 1


    def record_group_lstm_scores(self, predictions):
        # this function to record lstm scores into group attributes
        total_group_count = self.get_total_group_count()
        assert len(predictions) == total_group_count
        for i in range(total_group_count):
            self.groups[i].group_lstm_score = predictions[i]

class all_detections_one_view(all_detections):
    def __init__(self, folder_name, view_name):
        self.view_name = view_name
        super().__init__(folder_name)