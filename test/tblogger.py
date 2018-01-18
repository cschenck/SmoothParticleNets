"""
This file provides the class TBLogger (TensorBoard Logger) for creating
summary files that can then be viewed by TensorBoard. It can log the
following types of data:
    -scalars
    -images
    -histograms
    -3D scatter plots
    -3D grids
    -3D quiver plots (i.e., plots of arrows)
    -3D vector difference plots (comparing two different sets of 
     vectors for the same set of points).
This file requires tensforflow, matplotlib, scipy, and opencv.
"""

# Code referenced from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import cv2

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class TBLogger(object):
    """The TBLogger class. It takes a directory as an argument to its constructor
    and creates log files in that directory. Make sure to call flush periodically
    to ensure that the log file is written to.
    """
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        plt.ioff()
        self.fig = plt.figure()

    def __del__(self):
        self.flush()

    def flush(self):
        self.writer.flush()


    def scatter3d_summary(self, tag, step, locs, data=None,
            axlimits=None, titles=None, sized=False):
        """Log a 3D scatter plot.
            -tag: The string tag to associate with this log.
            -step: The timestep for this log.
            -locs: A Nx3 array of xyz locations to plot.
            -data: [Optional] A NxC array of values in [0, 1]. C plots
                   will be generated, and the color of the points in each
                   plot will be created from this array using the jet
                   colormap.
            -axlimits: [Optional] If specified, should be a 3x2 array of
                       the limits of the xyz axes.
            -titles: [Optional] A list of length C containing strings.
                     The strings will be the title of each plot.
            -sized: [Optional] If true, the values in data will also be
                    used to determine the size of each point.
        """
        self.fig.clf()
        if data is None:
            self.fig.set_size_inches(7, 5, forward=True)
            axs = [self.fig.add_subplot(111, projection='3d')]
        else:
            self.fig.set_size_inches(7*data.shape[-1], 5, forward=True)
            axs = [self.fig.add_subplot(1,data.shape[-1],i+1, projection='3d') 
                    for i in range(data.shape[-1])]
            colors = cv2.applyColorMap((1.0 - np.clip(data, 0, 1)*255).astype(np.uint8), 
                cv2.COLORMAP_JET).astype(np.float32)/255.0
        for i, ax in enumerate(axs):
            if data is None:
                ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2])
            else:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
                if sized:
                    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], c=colors[:, i, :],
                        s=data[:, i], norm=norm)
                else:
                    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], c=colors[:, i, :], 
                        norm=norm)
                if titles is not None:
                    ax.set_title(titles[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            if axlimits is not None:
                ax.set_xlim(*axlimits[0])
                ax.set_ylim(*axlimits[1])
                ax.set_zlim(*axlimits[2])
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape((1,) + self.fig.canvas.get_width_height()[::-1] + (3,))
        img = self._remove_empty_cols(img)
        self.image_summary(tag, img, step)

    def quiver3d_summary(self, tag, step, locs, vec,
            axlimits=None, titles=None):
        """Log a 3D quiver (i.e., arrow) plot.
            -tag: The string tag to associate with this log.
            -step: The timestep for this log.
            -locs: A Nx3 array of xyz locations to plot.
            -Vec: A NxCx3 array of vectors for each point. C
                  plots will be generated.
            -axlimits: [Optional] If specified, should be a 3x2 array of
                       the limits of the xyz axes.
            -titles: [Optional] A list of length C containing strings.
                     The strings will be the title of each plot.
        """
        self.fig.clf()
        self.fig.set_size_inches(7*vec.shape[-2], 5, forward=True)
        axs = [self.fig.add_subplot(1,vec.shape[-2],i+1, projection='3d') 
                for i in range(vec.shape[-2])]
        for i, ax in enumerate(axs):
            ax.quiver(locs[:, 0], locs[:, 1], locs[:, 2], vec[:, i, 0], 
                vec[:, i, 1], vec[:, i, 2])
            if titles is not None:
                ax.set_title(titles[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            if axlimits is not None:
                ax.set_xlim(*axlimits[0])
                ax.set_ylim(*axlimits[1])
                ax.set_zlim(*axlimits[2])
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape((1,) + self.fig.canvas.get_width_height()[::-1] + (3,))
        img = self._remove_empty_cols(img)
        self.image_summary(tag, img, step)

    def vecdiff_summary(self, tag, step, locs, vec1, vec2,
            axlimits=None, scale=1.0):
            """Log a 3D vector difference plot, that is, plot two sets of vectors
            for the same set of points alongside eachother so they may be easily
            compared. The vectors in vec1 are plotted green, the vectors in vec2
            are plotted blue, and the differences between the two are plotted in
            red.
            -tag: The string tag to associate with this log.
            -step: The timestep for this log.
            -locs: A Nx3 array of xyz locations to plot.
            -vec1: A Nx3 array of xyz vectors.
            -vec2: A Nx3 array of xyz vectors.
            -axlimits: [Optional] If specified, should be a 3x2 array of
                       the limits of the xyz axes.
            -scale: [Optional] Scale the magnitude of every vector by this value.
                    Useful when they are too small or large to be seen.
        """
        self.fig.clf()
        self.fig.set_size_inches(14, 10, forward=True)
        ax = self.fig.add_subplot(111, projection='3d')
        N = locs.shape[0]
        vec1 *= scale
        vec2 *= scale
        vec3 = vec1 - vec2
        locs = np.concatenate((locs, locs, locs + vec2), axis=0)
        colors = np.zeros((3*N, 4), dtype=np.float32)
        colors[:, 3] = 1.0
        colors[:N, 1] = 1.0
        colors[N:(2*N), 2] = 1.0
        colors[(2*N):, 0] = 1.0
        vec = np.concatenate((vec1, vec2, vec3), axis=0)
        ax.quiver(locs[:, 0], locs[:, 1], locs[:, 2], vec[:, 0], 
            vec[:, 1], vec[:, 2], colors=colors, arrow_length_ratio=0.0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if axlimits is not None:
            ax.set_xlim(*axlimits[0])
            ax.set_ylim(*axlimits[1])
            ax.set_zlim(*axlimits[2])
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape((1,) + self.fig.canvas.get_width_height()[::-1] + (3,))
        img = self._remove_empty_cols(img)
        self.image_summary(tag, img, step)

    def grid3d_summary(self, tag, step, grid, grid_lower, grid_steps, titles=None,
            draw_threshold=0.1, transparency_scale=1.0/16.0, shape=None, 
            color_norms=(0, 1)):
        """Log a 3D grid plot. All cells are plotted with transparency based on the
        absolute value of the value in the grid cell, with 0 being completely
        transparent and 1 being completely opaque. The color of each cell is based
        on jet colormap. Warning: this function is slow.
            -tag: The string tag to associate with this log.
            -step: The timestep for this log.
            -grid: A XxYxZxC array with values to be plotted. 
            -grid_lower: A length 3 tuple/list of the xyz location of the lower
                         corner of the grid.
            -grid_steps: A length 3 tuple/list of the xyz size of each grid cell.
            -titles: [Optional] A list of length C containing strings.
                     The strings will be the title of each plot.
            -draw_threshold: [Optional] If the alpha value for a grid cell is less
                             than this, don't draw it (for efficiency).
            -transparency_scale: [Optional] Multiply the alpha values by this
                                 amount. Can make viewing dense grids easier.
            -shape: [Optional] Instead of drawing all C plots in a row, specify
                    a length 2 tuple/list for the number of rows and columns.
            -color_norms: [Optional] The min and max value to use when scaling
                          the values in the grid cells to create the color scheme.
        """
        if shape is None:
            shape = (1, grid.shape[-1])
        row, col, dep = np.indices(grid.shape[:3])
        row = row.flatten()
        col = col.flatten()
        dep = dep.flatten()
        x = row*grid_steps[0] + grid_lower[0]
        y = col*grid_steps[1] + grid_lower[1]
        z = dep*grid_steps[2] + grid_lower[2]
        data = grid[row, col, dep]
        colors = cv2.applyColorMap(
            (255 - np.clip((data - color_norms[0])/(color_norms[1] - color_norms[0]), 0, 1)*255)
                .astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)/255.0
        self.fig.clf()
        self.fig.set_size_inches(7*shape[1], 5*shape[0], forward=True)
        axs = [self.fig.add_subplot(shape[0],shape[1],i+1, projection='3d') 
                for i in range(data.shape[-1])]
        for i, ax in enumerate(axs):
            alpha = np.clip(np.abs(data[:, i]), 0, 1)
            idxs = np.where(alpha > draw_threshold)[0]
            alpha = alpha[idxs]*transparency_scale
            c = np.concatenate((colors[idxs, i, :], np.expand_dims(alpha, -1)), axis=-1)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
            ax.scatter(x[idxs], y[idxs], z[idxs], c=c, edgecolors='face', norm=norm)
            if titles is not None:
                ax.set_title(titles[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlim((grid_lower[0], grid_steps[0]*grid.shape[0]))
            ax.set_ylim((grid_lower[1], grid_steps[1]*grid.shape[1]))
            ax.set_zlim((grid_lower[2], grid_steps[2]*grid.shape[2]))
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape((1,) + self.fig.canvas.get_width_height()[::-1] + (3,))
        img = self._remove_empty_cols(img)
        self.image_summary(tag, img, step)


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
            scipy.misc.toimage(img).save(s, format="jpeg")

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

    def _remove_empty_cols(self, img):
        axes = list(range(len(img.shape)))
        axes.remove(2)
        cols = np.where(img.sum(axis=(0, 1, 3)) < img.shape[1]*img.shape[-1]*255)[0]
        return img[:, :, cols, :]

