import numpy as np
from . import mtcnn
from PIL import Image
from scipy import misc


class FaceAligner():

    def __init__(self, allow_low_threshold=True, fallback_centercrop=True, crop_size=(112, 112), device='cuda:0'):
        # allow_low_threshodl: adjust MTCNN Threshold to allow more faces
        # fallback_centercrop: when MTCNN fails, perform center crop
        self.allow_low_threshold = allow_low_threshold
        self.fallback_centercrop = fallback_centercrop
        self.mtcnn_model = mtcnn.MTCNN(device=device, crop_size=crop_size, low_threshold=allow_low_threshold)
        self.crop_size = crop_size

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


    def align(self, image):
        if image is None:
            return None
        rgb_img = self.get_rgb_image(image)

        # find face
        bboxes, faces = self.mtcnn_model.align_multi(rgb_img, limit=1)

        if len(faces) > 0:  # get top1 face
            face = faces[0]
        else:
            if self.fallback_centercrop:
                face = self.center_crop(rgb_img)
            else:
                face = None

        return face

    def center_crop(self, rgb_img: Image.Image) -> Image.Image:
        # center crop
        assert self.crop_size[0] == self.crop_size[1], 'non square center crop not implemented yet'
        scaled = np.array(rgb_img.resize(size=(196,196)))
        sz1 = scaled.shape[1] // 2
        sz2 = self.crop_size[0] // 2
        face = scaled[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
        assert face.shape[0] == self.crop_size[0]
        assert face.shape[1] == self.crop_size[0]
        face = Image.fromarray(face)
        return face
