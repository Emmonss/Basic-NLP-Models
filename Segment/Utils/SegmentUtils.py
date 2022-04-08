
import numpy as np
import re
'''
'''
class StaticSegment:
    def __init__(self):
        super().__init__(self)

    @staticmethod
    def get_precision(y_gold,y_true):
        return float(len(y_gold) / len(y_true))

    @staticmethod
    def get_recall(y_gold,y_pred):
        return float(len(y_gold)/len(y_pred))

    @staticmethod
    def get_fl(precision,recall):
        return float(2*precision*recall/precision+recall)

    @staticmethod
    def get_correct(y_true_1,y_pred_1):
        res = []
        for word in y_pred_1:
            if word in y_true_1:
                res.append(word)
        return res

    @staticmethod
    def get_word_accuracy_single(list1,list2):
        assert len(list1) == len(list2), 'the length is not equal'
        acc = 0
        for index in enumerate(list1):
            if list1[index]==list2[index]:
               acc+=1
        return acc,len(list2)

    @staticmethod
    def to_region(segmentation):
        region = []
        start = 0
        seg = ''
        for word in segmentation:
            seg+='{} '.format(word)

        for word in re.compile("\\s+").split(seg.strip()):
            end = start + len(word)
            region.append((start, end))
            start = end
        return region

    @staticmethod
    def get_oov_iv(y_true,y_pred,word_dict):
        assert len(y_pred) == len(y_true), 'the length is not equal'

        oov,oov_r,iv,iv_r=0,0,0,0
        for index in range(len(y_true)):
            y_true_region = StaticSegment.to_region(y_true[index])
            y_pred_region = StaticSegment.to_region(y_pred[index])

            text = ""
            for item in y_true[index]:
                text+=item

            for (start, end) in y_true_region:
                word = text[start: end]
                if word in word_dict:
                    iv += 1
                else:
                    oov += 1

            for (start, end) in y_true_region & y_pred_region:
                word = text[start: end]
                if word in word_dict:
                    iv_r += 1
                else:
                    oov_r += 1
        return float(iv_r/iv),float(oov_r/oov)

    @staticmethod
    def get_word_accuracy(y_true,y_pred):
        assert len(y_pred) == len(y_true), 'the length is not equal'
        acc_count,all_count=0,0
        for index in range(len(y_true)):
            acc_count_1,all_count_1 =StaticSegment.get_word_accuracy_single(y_true[index],y_pred[index])
            acc_count+=acc_count_1
            all_count+=all_count_1
        return acc_count,all_count,float(acc_count/all_count)

    @staticmethod
    def get_segment_pred(y_true,y_pred):
        assert len(y_pred)==len(y_true),'the length is not equal'
        true_all_count = []
        pred_all_count = []
        correct_all_count =[]
        for index in range(len(y_true)):
            correct_1 = StaticSegment.get_correct(y_pred[index],y_true[index])
            true_all_count+=y_true[index]
            pred_all_count+=y_pred[index]
            correct_all_count+=correct_1
        precision = StaticSegment.get_precision(correct_all_count,true_all_count)
        recall = StaticSegment.get_precision(correct_all_count, pred_all_count)
        f1_score = StaticSegment.get_fl(precision,recall)

        return f1_score,recall

    @staticmethod
    def get_pred_main(true_csv,pred_csv):
        pass






if __name__ == '__main__':
    pass