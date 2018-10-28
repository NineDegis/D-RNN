import tensorflow as tf
import numpy as np
import os
import errno
from enum import Enum
from datetime import datetime


# TODO(skrudtn): Make tensor board can be accessed from the outside
class TensorBoardLogger(object):
    class Mode(Enum):
        SCALAR = 1
        HISTOGRAM = 2

    def __init__(self, log_dir):
        self.make_dir(log_dir)
        log_path = self.get_dir_name(log_dir)
        self.make_dir(log_path)
        self.writer = tf.summary.FileWriter(log_path)

    def summary(self, mode, tag, values, step, bins=100):
        """Log a scalar variable and histogram of the tensor of values.

        Args:
            mode: Enum to determine mode.
            tag: The name of value data.
            values: Data to log.
            step: Number of times data is processed.
            bins: bin 이 int 지정된 범위(default = 100)의 동일한 컨테이너 수를 정의한다.
                bin 이 sequence 인 경우 오른쪽 edge 를 지정하여 non-uniform bin width 를 허용한다.

                Example:
                    [ 0.5, 1.1, 1.3, 2.2, 2.9, 2.99 ]의 숫자 배열이 있을 때, bins 를 3으로 둔다면
                    0 - 1 bin: (0.5)
                    1 - 2 bin: (1.1, 1.3)
                    2 - 3 bin: (2.2, 2.9, 2.99)
                    위와같은 3개의 bin 으로 만들어 진다.
                See: https://www.tensorflow.org/guide/tensorboard_histograms
        """
        if self.Mode.SCALAR == mode:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=values)])
            self.writer.add_summary(summary, step)
        elif self.Mode.HISTOGRAM == mode:
            # Create a histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill the fields of the histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Drop the start of the first bin, due to "len(bin_edges) - 1 == len(counts)"
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)
            self.writer.flush()

    # TODO(kyungsoo): Modify it to get more general parameters
    def log(self, loss_avg, accuracy_avg, model_named_parameters, cur_epoch):
        """ Function to receive information to log from trainer.

        Args:
            loss_avg: Average of losses per epoch.
            accuracy_avg: Average of accuracy per epoch
            model_named_parameters: The named_parameter of the model.
            cur_epoch: Current epoch.
        """
        info = {"loss": loss_avg, "accuracy": accuracy_avg}
        for tag, value in info.items():
            self.summary(self.Mode.SCALAR, tag, value, cur_epoch + 1)
        for tag, value in model_named_parameters:
            tag = tag.replace('.', '/')
            self.summary(self.Mode.HISTOGRAM, tag, value.data.cpu().numpy(), cur_epoch + 1)
            try:
                self.summary(self.Mode.HISTOGRAM, tag + '/grad', value.grad.data.cpu().numpy(), cur_epoch + 1)
                # print(tag, 'is fine')
            except AttributeError:
                # print(tag, 'is the error')
                pass

    @staticmethod
    def get_dir_name(log_dir):
        """Create a folder name

        log dir format - "Model Name"\"number"_"YYYY-MM-DD"T"hh.mm.ss"
        Args:
            log_dir: Folder in which to save logs.
        Return:
            str: dir_name
        """
        if len(os.listdir(log_dir)) == 0:
            run = 1
        else:
            dirs = os.listdir(log_dir)
            no_list = []
            for directory in dirs:
                no_list.append(int(directory.split("_")[0]))
            no_list.sort()
            run = no_list[len(no_list) - 1] + 1

        now = datetime.now()
        date = str(now.date())
        time = str(now.time().hour) + "." + str(now.time().minute) + "." + str(now.time().second)

        log_path = os.path.join(log_dir, '%d_%s' % (run, date + "T" + time))

        return log_path

    @staticmethod
    def make_dir(path):
        """Create a folder if it does not exist at that location, or an error

        Args:
            path: Where to create the folder.
        """
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        except Exception as e:
            raise e
