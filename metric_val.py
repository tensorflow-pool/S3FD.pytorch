import os

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np


class MApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    ovp_thresh : float
        overlap threshold for TP
    use_difficult : boolean
        use difficult ground-truths if applicable, otherwise just ignore
    class_names : list of str
        optional, if provided, will print out AP for each class
    pred_idx : int
        prediction index in network output list
    roc_output_path
        optional, if provided, will 
        a ROC graph for each class
    tensorboard_path
        optional, if provided, will save a ROC graph to tensorboard
    """

    def __init__(self, ovp_thresh=0.35, use_difficult=True, class_names=None, pred_idx=0, roc_output_path=None, tensorboard_path=None):
        super(MApMetric, self).__init__('mAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = class_names + ['mAP', 'fp_200', 'fp_1000', 'max_recall']
            self.num = num + 4
        self.reset()
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)
        self.roc_output_path = roc_output_path
        self.tensorboard_path = tensorboard_path

    def save_roc_graph(self, recall=None, prec=None, cls="", path=None, ap=None):
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path = os.path.join(path, 'roc_' + cls)
        if os.path.exists(plot_path):
            os.remove(plot_path)
        fig = plt.figure()
        plt.title(cls)
        plt.plot(recall, prec, 'b', label='AP = %0.2f' % ap)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(plot_path)
        plt.close(fig)

    def save_fddb_roc_graph(self, recall=None, fp=None, cls="", path=None):
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path = os.path.join(path, 'fddb_roc_' + cls)
        if os.path.exists(plot_path):
            os.remove(plot_path)
        fig = plt.figure()
        plt.title(cls)
        plt.plot(fp, recall, 'b', label='fddb roc')
        plt.legend(loc='lower right')
        plt.xlim([0, 1000])
        plt.ylim([0, 1])
        plt.ylabel('recall')
        plt.xlabel('false postive')
        plt.savefig(plot_path)
        plt.close(fig)

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self.records = dict()
        self.counts = dict()

    def summary(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self.update_roc()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s' % (self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                      for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def update(self, labels, preds, thresh):
        """
        Update internal records. This function now only update internal buffer,
        sum_metric and num_inst are updated in _update() function instead when
        get() is called to return results.

        Params:
        ----------
        labels: [batch, num, [4-d, cls]]
        preds: [batch, num, [conf, 4-d]]
        """

        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                  (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        # independant execution for each image
        for i in range(len(labels)):
            # get as numpy arrays
            label = labels[i].cpu().numpy()
            pred = preds[i]
            #             print("label", label)
            #             print("pred", pred)
            # calculate for each class
            cid = 1
            keep_index = np.where(pred[:, 0] > thresh)[0]
            dets = pred[keep_index, :]
            # sort by score, desceding
            dets[dets[:, 0].argsort()[::-1]]
            records = np.hstack((dets[:, 0][:, np.newaxis], np.zeros((dets.shape[0], 1))))
            # ground-truths
            gts = label
            if gts.size > 0:
                found = [False] * gts.shape[0]
                for j in range(dets.shape[0]):
                    # compute overlaps
                    ious = iou(dets[j, 1:], gts[:, :4])
                    ovargmax = np.argmax(ious)
                    ovmax = ious[ovargmax]
                    if ovmax > self.ovp_thresh:
                        if (not self.use_difficult and
                                gts.shape[1] >= 6 and
                                gts[ovargmax, 5] > 0):
                            pass
                        else:
                            if not found[ovargmax]:
                                records[j, -1] = 1  # tp
                                found[ovargmax] = True
                            else:
                                # duplicate
                                records[j, -1] = 2  # fp
                    else:
                        records[j, -1] = 2  # fp
            else:
                # no gt, mark all fp
                records[:, -1] = 2

            # ground truth count
            if (not self.use_difficult and gts.shape[1] >= 6):
                gt_count = np.sum(gts[:, 5] < 1)
            else:
                gt_count = gts.shape[0]

            # now we push records to buffer
            # first column: score, second column: tp/fp
            # 0: not set(matched to difficult or something), 1: tp, 2: fp
            records = records[np.where(records[:, -1] > 0)[0], :]
            if records.size > 0:
                self._insert(cid, records, gt_count)

    def update_roc(self):
        """ update num_inst and sum_metric """
        aps = []
        recall_200 = 0
        recall_1000 = 0
        recall_max = 0
        for k, v in self.records.items():
            recall, prec, tp, fp, pos = self._recall_prec(v, self.counts[k])
            ap = self._average_precision(recall, prec)
            recall_200 = self.recall_at(recall, fp, 200)
            recall_1000 = self.recall_at(recall, fp, 1000)
            recall_max = self.recall_at(recall, fp, None)
            if self.roc_output_path is not None:
                self.save_roc_graph(recall=recall, prec=prec, cls="face", path=self.roc_output_path, ap=ap)
                self.save_fddb_roc_graph(recall=recall, fp=fp, cls="face", path=self.roc_output_path)
            aps.append(ap)
            if self.num is not None and k < (self.num - 1):
                self.sum_metric[k] = ap
                self.num_inst[k] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.mean(aps)
        else:
            cls_count = 1
            self.num_inst[cls_count] = 1
            self.sum_metric[cls_count] = np.mean(aps)
            self.num_inst[cls_count + 1] = 1
            self.sum_metric[cls_count + 1] = recall_200
            self.num_inst[cls_count + 2] = 1
            self.sum_metric[cls_count + 2] = recall_1000
            self.num_inst[cls_count + 3] = 1
            self.sum_metric[cls_count + 3] = recall_max

    def recall_at(self, recall, fp, fp_at):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        if fp_at is None:
            return np.max(recall)
        if np.sum(fp <= fp_at) == 0:
            return 0
        else:
            return np.max(recall[fp <= fp_at])

    def _recall_prec(self, record, count):
        """ get recall and precision from internal records """
        record = np.delete(record, np.where(record[:, 1].astype(int) == 0)[0], axis=0)
        sorted_records = record[record[:, 0].argsort()[::-1]]
        tp = np.cumsum(sorted_records[:, 1].astype(int) == 1)
        fp = np.cumsum(sorted_records[:, 1].astype(int) == 2)
        if count <= 0:
            recall = tp * 0.0
        else:
            recall = tp / float(count)
        prec = tp.astype(float) / (tp + fp)
        return recall, prec, tp, fp, count

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _insert(self, key, records, count):
        """ Insert records according to key """
        if key not in self.records:
            assert key not in self.counts
            self.records[key] = records
            self.counts[key] = count
        else:
            self.records[key] = np.vstack((self.records[key], records))
            assert key in self.counts
            self.counts[key] += count


class VOC07MApMetric(MApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset """

    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap