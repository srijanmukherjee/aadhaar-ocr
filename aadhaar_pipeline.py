import base64
from dataclasses import dataclass
import datetime
from io import BytesIO
import re
import sys
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

import cv2
from loguru import logger
import pytesseract
from ultralytics.engine.model import Model
from ultralytics.engine.results import Boxes
from pipeline import LinearPipeline, Map, Pipeline, PipelineStep, Reduce

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {thread.name: <20} | <level>{message}</level>"
logger.configure(handlers=[dict(sink=sys.stderr, format=LOG_FORMAT)])

@dataclass
class Feature:
    id: str
    conf: float
    rect: Tuple[int, int, int, int]
    image: Image
    
class LoadImageStep(PipelineStep):
    def execute(self, _, data: str) -> Image:
        return Image.open(BytesIO(base64.b64decode(data)))

class FeatureExtractionStep(PipelineStep):
    model: Model

    def __init__(self, model: Model):
        self.model = model
        self.internal_class_mapping = {
            'adhaar_number': 'aadhaar_number',
            'date_of_birth-year': 'date_of_birth'
        }

    def execute(self, _, image: Image) -> Dict[str, Feature]:
        predictions = self.model(image, verbose=False)

        if len(predictions) == 0:
            raise Exception("no feature found")
        
        if len(predictions) > 1:
            raise Exception("more than one id found")
        
        prediction = predictions[0]
        classes = self.map_model_classes_to_internal_classes(prediction.names)
        boxes: Boxes = prediction.boxes
        output = {}

        for result in boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            cls = int(cls)
            cls = classes[cls]
            output[cls] = Feature(cls, conf, (int(x1), int(y1), int(x2), int(y2)), image)
            logger.info(f'[FeatureExtractionStep] feature: {cls}, conf: {conf}')

        logger.info(f'[FeatureExtractionStep] extracted features: {output.keys()}')

        return output
    
    def map_model_classes_to_internal_classes(self, classes: dict) -> dict:
        return { k: self.internal_class_mapping.get(v, v) for k, v in classes.items() }
    
class FilterFeatureByThresholdStep(PipelineStep):
    def __init__(self, 
                 feature_threshold_key = 'feature_threshold', 
                 default_threshold_key='default_feature_threshold', 
                 default_threshold=0.0):
        self.feature_threshold_key = feature_threshold_key
        self.default_threshold_key = default_threshold_key
        self.default_threshold = default_threshold

    def execute(self, config: Dict, data: Dict[str, Feature]) -> Dict[str, Feature]:
        return { k: v for k, v in data.items() if v.conf >= self.get_feature_threshold(config, v.id) }
    
    def get_feature_threshold(self, config: Dict, id: str) -> float:
        thresholds = config.get(self.feature_threshold_key, {})
        default_threshold = config.get(self.default_threshold_key, self.default_threshold)
        return thresholds.get(id, default_threshold)

class FeatureOpticalCharacterRecognitionStep(PipelineStep):
    preprocessing_pipeline: Pipeline
    preview: bool
    
    def __init__(self, preprocessing_pipeline: Pipeline, preview = False):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.preview = preview

    def execute(self, config: Dict, feature: Feature) -> str:
        logger.info(f'[FeatureOpticalCharacterRecognitionStep] {feature.id}')
        
        x1, y1, x2, y2 = feature.rect

        image = cv2.cvtColor(np.array(feature.image), cv2.COLOR_RGB2BGR)
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        images = self.preprocessing_pipeline.execute(config, {
            'id': feature.id,
            'image': image
        })

        if self.preview:
            for image in images:
                cv2.imwrite(f'{feature.id}-{datetime.datetime.now().timestamp()}.png', image)

        if feature.id == 'aadhaar_number':
            config = '--psm 7 -c tessedit_char_whitelist=0123456789' # single line text
        elif feature.id == 'address':
            config = '--psm 6' # block of text
        elif feature.id == 'name':
            config = '--psm 7'
        else:
            config = ''

        for image in images:            
            ocr_value: str = pytesseract.image_to_string(image, lang='eng', config=config)
            logger.debug(f'[FeatureOpticalCharacterRecognitionStep] feature: {feature.id}, value: {ocr_value}')
            if ocr_value.isalnum():
                break

        return ocr_value
    
class SanitizeValueStep(PipelineStep):
    def execute(self, _, data: str) -> str:
        return data.replace('\n', ' ').strip()

class AadhaarReducerStep(PipelineStep):
    def execute(self, _, data: Tuple[Dict, Dict]) -> any:
        prev, curr = data

        if prev is None:
            return curr
        
        curr_aadhaar_number = curr.get('aadhaar_number', None)
        prev_aadhaar_number = prev.get('aadhaar_number', None)
        if prev_aadhaar_number != curr_aadhaar_number:
            raise Exception(f"aadhaar numbers do not match: {curr_aadhaar_number} != {prev_aadhaar_number}")
        
        return prev | curr
    
class FeaturePreprocessingStep(PipelineStep):
    def execute(self, _: Dict, data: any) -> List[cv2.Mat]:
        if data['id'] == 'aadhaar_number':
            return [self.preprocess_aadhaar_number(data['image']), data['image']]
        elif data['id'] == 'address':
            return [self.preprocess_adress(data['image'])]
        elif data['id'] == 'name':
            return [self.preprocess_name(data['image'])]
        return [data['image']]
    
    def preprocess_aadhaar_number(self, image: cv2.Mat) -> cv2.Mat:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        return image
    
    def preprocess_adress(self, image: cv2.Mat) -> cv2.Mat:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
        # # _,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image
    
    def preprocess_name(self, image: cv2.Mat) -> cv2.Mat:
        # Font thinning
        # image = cv2.bitwise_not(image)
        # kernel = np.ones((1,1), np.uint8)
        # image = cv2.erode(image, kernel, iterations=1)
        # image = cv2.bitwise_not(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        return image

class AadhaarFieldTranslationStep(PipelineStep):
    def execute(self, _, data: any) -> any:
        address: str = data.get('address', None)
        if address:
            address_prefix = 'address:'
            if address.lower().startswith(address_prefix):
                data['address'] = address[len(address_prefix):].strip()
        
        gender: str = data.get('gender', None)
        if gender:
            data['gender'] = gender.lower()

        aadhaar_number = data.get('aadhaar_number', None)
        if aadhaar_number:
            data['aadhaar_number'] = aadhaar_number.replace(' ', '')
        
        return data

class AadhaarValidationStep(PipelineStep):
    DEFAULT_REQUIRED_FIELDS = ['aadhaar_number', 'name', 'gender', 'date_of_birth']
    
    def execute(self, config: Dict, data: Dict[str, str]) -> Dict[str, str]:
        required_fields = config.get('required_fields', AadhaarValidationStep.DEFAULT_REQUIRED_FIELDS)

        for field in required_fields:
            if field not in data or len(data[field]) == 0:
                raise Exception(f"{field} is missing")
        
        return data

class GetCardStep(PipelineStep):
    def execute(self, config: Dict, image: Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = self.get_config(config, 'thresh', 120)
        maxval = self.get_config(config, 'maxval', 255)
        _, thresh_image = cv2.threshold(gray_image, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)

        kernel_size = self.get_config(config, 'blur_kernel_size', 15)
        blurred_image = cv2.GaussianBlur(thresh_image, (kernel_size, kernel_size), 0)

        contour_approximation_mode = self.get_config(config, 'contour_approximation_mode', cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(blurred_image, cv2.RETR_LIST, contour_approximation_mode)

        max_bbox = (0, 0, 0, 0)
        max_area = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                bbox = cv2.boundingRect(contour)
                area = bbox[2] * bbox[3] # width * height
                if area > max_area:
                    max_bbox = bbox
                    max_area = area

        x, y, w, h = max_bbox
        area = w * h
        image_area = image.shape[0] * image.shape[1]
        area_cover_percentage = self.get_config(config, 'area_cover_percentage', 0.2)
        if area < image_area * area_cover_percentage:
            cropped_image = image
        else:
            cropped_image = image[y : y + h, x : x + w]

        auto_rotate_enabled = self.get_config(config, 'auto_rotate', True)
        if auto_rotate_enabled:
            cropped_image = self.rotate(cropped_image)

        cv2.imwrite(f'{datetime.datetime.now().timestamp()}.png', cropped_image)

        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    def rotate(self, image, center = None, scale = 1.0):
        try:
            angle = 360 - int(re.search(r'(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
            (h, w) = image.shape[:2]

            if center is None:
                center = (w / 2, h / 2)
            
            return cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, scale), (w, h))
        except Exception:
            logger.warning('[GetCardStep] Encountered an exception while auto rotating image, using original image')
            return image

    def get_config(self, config: dict, key: str, default: any) -> any:
        config = config.get('get_card_step', {})
        return config.get(key, default)


def build_aadhaar_pipeline(feature_extraction_model: Model, preview_ocr = False):
    # TODO: integrate the preprocessnig pipeline into the main pipeline
    preprocessing_pipeline = LinearPipeline([
        FeaturePreprocessingStep(),
    ])
    
    return LinearPipeline([
        Map([
            LoadImageStep(),
            GetCardStep(),
            FeatureExtractionStep(feature_extraction_model),
            FilterFeatureByThresholdStep(),
            Map([
                FeatureOpticalCharacterRecognitionStep(preprocessing_pipeline, preview=preview_ocr),
                SanitizeValueStep()
            ], parallel=True)
        ]),
        Reduce(AadhaarReducerStep()),
        AadhaarFieldTranslationStep(),
        AadhaarValidationStep()
    ])
