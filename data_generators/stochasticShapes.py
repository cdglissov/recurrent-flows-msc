'''Stochastic moving shapes'''

import math
import numpy as np
import matplotlib.pyplot as plt


class VideoStochasticShapes():
    """Shapes moving in a stochastic way."""
    def __init__(self, n_videos=10):
        self.n_videos = n_videos

    @property
    def frame_height(self):
        return 64

    @property
    def frame_width(self):
        return 64

    @property
    def total_number_of_frames(self):
        return self.n_videos * self.video_length

    @property
    def video_length(self):
        return 5

    @staticmethod
    def get_circle(x, y, z, c, s):
        """Draws a circle with center(x, y), color c, size s and z-order of z."""
        cir = plt.Circle((x, y), s, fc=c, zorder=z)
        return cir

    @staticmethod
    def get_rectangle(x, y, z, c, s):
        """Draws a rectangle with center(x, y), color c, size s and z-order of z."""
        rec = plt.Rectangle((x - s, y - s), s * 2.0, s * 2.0, fc=c, zorder=z)
        return rec

    @staticmethod
    def get_triangle(x, y, z, c, s):
        """Draws a triangle with center (x, y), color c, size s and z-order of z."""
        points = np.array([[0, 0], [s, s * math.sqrt(3.0)], [s * 2.0, 0]])
        tri = plt.Polygon(points + [x - s, y - s], fc=c, zorder=z)
        return tri

    def generate_stochastic_shape_instance(self):
        """Yields one video of a shape moving to a random direction.
           The size and color of the shapes are random but
           consistent in a single video. The speed is fixed.
        Raises:
           ValueError: The frame size is not square.
        """
        if self.frame_height != self.frame_width or self.frame_height % 2 != 0:
            raise ValueError("Generator only supports square frames with even size.")

        lim = 10.0
        direction = np.array([[+1.0, +1.0],
                              [+1.0, +0.0],
                              [+1.0, -1.0],
                              [+0.0, +1.0],
                              [+0.0, -1.0],
                              [-1.0, +1.0],
                              [-1.0, +0.0],
                              [-1.0, -1.0]
                              ])

        sp = np.array([lim / 2.0, lim / 2.0])
        rnd = np.random.randint(len(direction))
        di = direction[rnd]

        colors = ["b", "g", "r", "c", "m", "y"]
        color = np.random.choice(colors)

        shape = np.random.choice([
            VideoStochasticShapes.get_circle,
            VideoStochasticShapes.get_rectangle,
            VideoStochasticShapes.get_triangle])
        speed = 1.0

        size = np.random.uniform(0.5, 1.5)

        back_color = str(0.0)
        plt.ioff()

        xy = np.array(sp)

        for _ in range(self.video_length):
            fig = plt.figure()
            fig.set_dpi(self.frame_height // 2)
            fig.set_size_inches(2, 2)
            ax = plt.axes(xlim=(0, lim), ylim=(0, lim))

            # Background
            ax.add_patch(VideoStochasticShapes.get_rectangle(
                0.0, 0.0, -1.0, back_color, 25.0))

            # Foreground
            ax.add_patch(shape(xy[0], xy[1], 0.0, color, size))

            plt.axis("off")
            plt.tight_layout(pad=-2.0)
            fig.canvas.draw()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = np.copy(np.uint8(image))

            plt.close()
            xy += speed * di

            yield image

    def generate_samples(self):
        counter = 0
        done = False
        while not done:
            for frame_number, frame in enumerate(
                    self.generate_stochastic_shape_instance()):
                if counter >= self.total_number_of_frames:
                    done = True
                    break

                yield {"frame": frame, "frame_number": [frame_number]}
                counter += 1
               
# TODO: allow for storing data, .h5 file is the better way to go. Make another function to use the data file and store
# it into dataloaders
