from . import fastface as ff
from PIL import Image
import numpy as np
import cv2

class FaceDetector():
    def __init__(self, device='cuda:0', output_shape=(256,256), model='lffd_slim'):
        if model == 'lffd_original':
            self.model = ff.FaceDetector.from_pretrained("lffd_original").eval()
            self.input_size = (640,640)  # width, height (x,y)
        elif model == 'lffd_slim':
            self.model = ff.FaceDetector.from_pretrained("lffd_slim").eval()
            self.input_size = (480,480)  # width, height (x,y)
        else:
            raise ValueError('not a correct model')

        self.model.to(device)
        self.output_shape = output_shape  # width, height (x,y)


    def get_rgb_image(self, image):
        if isinstance(image, str):
            rgb_img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # nd array is BGR. so change to RGB
            rgb_img = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        elif isinstance(image, Image.Image):
            rgb_img = image.convert('RGB')
        else:
            raise ValueError('not a correct type')
        return rgb_img

    def pad_bbox(self, bbox, padding_ratio, image_shape):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        pad_x = padding_ratio * width
        pad_y = padding_ratio * height
        xmin, ymin, xmax, ymax = xmin-pad_x, ymin-pad_y, xmax+pad_x, ymax+pad_y
        return (max(xmin, 0), max(ymin, 0), min(image_shape[1], xmax), min(image_shape[0], ymax))


    def detect(self, image) -> Image.Image:
        rgb_img = self.get_rgb_image(image)
        rgb_img_np = np.array(rgb_img)
        rgb_img_np_resized = cv2.resize(rgb_img_np, self.input_size)
        preds, = self.model.predict(rgb_img_np_resized, det_threshold=.8, iou_threshold=.4)

        if preds['boxes']:
            # convert bbox scale back to before resize
            xmin, ymin, xmax, ymax = preds['boxes'][0]  # [xmin, ymin, xmax, ymax]
            xscale = rgb_img_np.shape[1] / self.input_size[0]
            yscale = rgb_img_np.shape[0] / self.input_size[1]
            rescaled_bbox = [xmin*xscale, ymin*yscale, xmax*xscale, ymax*yscale]
            # pad bbox
            padded_bbox = self.pad_bbox(rescaled_bbox, padding_ratio=0.1, image_shape=rgb_img_np.shape)
            xmin, ymin, xmax, ymax = [int(i) for i in padded_bbox]

            # crop
            cropped_rgb_array = rgb_img_np[ymin:ymax, xmin:xmax]
            cropped_rgb_array = cv2.resize(cropped_rgb_array, self.output_shape)
            cropped = Image.fromarray(cropped_rgb_array)
        else:
            cropped = None

        return cropped