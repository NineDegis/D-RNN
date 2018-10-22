import tensorflow as tf
import numpy as np
import scipy.misc
import os
import errno

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        ensure_dir(log_dir)
        if len(os.listdir(log_dir)) == 0:
            run = 1
        else:
            run = int(os.listdir(log_dir)[len(os.listdir(log_dir))-1])+1

        self.log_dir = os.path.join(log_dir, '%d' % run)
        ensure_dir(log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def add_scalar(self, loss_avg, accuracy_avg, cur_epoch):
        info = {"loss": loss_avg, "accuracy": accuracy_avg}
        for tag, value in info.items():
            self.scalar_summary(tag, value, cur_epoch + 1)

    def add_histogram(self, model_named_parameters, cur_epoch):
        for tag, value in model_named_parameters:
            tag = tag.replace('.', '/')
            self.histo_summary(tag, value.data.cpu().numpy(), cur_epoch + 1)
            self.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), cur_epoch + 1)

def ensure_dir(file_path):
    try:
        os.makedirs(file_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
