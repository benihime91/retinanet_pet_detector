import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pytorch_retinanet.retinanet.utilities import ifnone

from .data_utils import _get_logger

# Turn interactive plotting off
plt.ioff()


class Visualizer:
    def __init__(self, class_names: Union[Dict[int, str], List[str]], logger=None):
        self.c_names = class_names
        self.colors = np.array(
            [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            dtype=np.float32,
        )
        if logger is None:
            self.logger = _get_logger(__name__)
        else:
            self.logger = logger
            self.logger.name = __name__

        self.logger.info("visualizer initialized")

    def _get_color(self, c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * self.colors[i][c] + ratio * self.colors[j][c]

        return int(r * 255)

    def draw_bboxes(
        self,
        img: Union[np.array, str],
        boxes: np.array,
        classes: Optional[Union[np.array, List]] = None,
        scores: Optional[Union[np.array, List]] = None,
        figsize: Tuple[int, int] = None,
        save: bool = False,
        save_dir: str = "outputs",
        show: bool = True,
        fname: str = "res.png",
        color=None,
        return_fig: bool = False,
    ):

        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
            img = np.array(img)

        # Create a figure and plot the image
        # NB: use smaller size beacuse matplotlib takes long time to render large images
        sz = ifnone(figsize, (15, 15))
        fig, a = plt.subplots(1, 1, figsize=sz)
        a.imshow(img)

        scores = ifnone(scores, np.repeat(1.0, axis=0, repeats=len(boxes)))
        self.logger.info(f"Found {len(boxes)} bounding box(s) on the given image")

        # Plot the bounding boxes and corresponding labels on top of the image
        for i in range(len(boxes)):
            # Get the ith bounding box
            box = boxes[i]
            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            rgb = (1, 0, 0)

            if classes is not None:
                cls_id = classes[i]
                cls_conf = scores[i]
                num_classes = len(self.c_names)
                offset = cls_id * 123457 % num_classes
                red = self._get_color(2, offset, num_classes) / 255
                green = self._get_color(1, offset, num_classes) / 255
                blue = self._get_color(0, offset, num_classes) / 255

                # If a color is given then set rgb to the given color instead
                if color is None:
                    rgb = (red, green, blue)
                else:
                    rgb = color

            width_x = x2 - x1
            width_y = y1 - y2

            # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
            # lower-left corner of the bounding box relative to the size of the image.
            c1 = (x1, y2)
            w1 = width_x
            w2 = width_y
            rect = patches.Rectangle(c1, w1, w2, lw=4, edgecolor=rgb, facecolor="none")
            # Draw the bounding box on top of the image
            a.add_patch(rect)

            # if classes are given plot the classes and the confidences
            if classes is not None:
                # Create a string with the object class name and the corresponding object class probability
                conf_tx = self.c_names[cls_id] + ": {:.1f}".format(cls_conf)
                # Define x and y offsets for the labels
                lxc = (img.shape[1] * 0.266) / 100
                lyc = (img.shape[0] * 1.180) / 100

                # Draw the labels on top of the image
                c1 = x1 + lxc
                c2 = y1 - lyc
                bb = dict(facecolor=rgb, edgecolor=rgb, alpha=0.8)
                a.text(c1, c2, conf_tx, fontsize=12, color="k", bbox=bb)

        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        plt.axis("off")

        if save:
            os.makedirs(save_dir, exist_ok=True)
            pth = os.path.join(save_dir, fname)
            plt.savefig(pth, bbox_inches="tight", pad_inches=0,)
            self.logger.info(f"Results saved to {os.path.join(save_dir, fname)}")

        if show:
            plt.show()

        if return_fig:
            return fig
