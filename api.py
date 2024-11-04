from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from aadhaar_pipeline import build_aadhaar_pipeline

model = YOLO("./model/best.pt", verbose=False)
aadhaar_pipeline = build_aadhaar_pipeline(model, preview_ocr=True)

base_aadhaar_pipieline_config = {
    'feature_threshold': {
        'aadhaar_number': 0.3,
        'name': 0.7,
        'date_of_birth': 0.5,
        'address': 0.8,
        'gender': 0.5
    },
    'required_fields': ['aadhaar_number'],
    'get_card_step': {
        'auto_rotate': False
    }
}

app = FastAPI()

class AadhaarOcrRequest(BaseModel):
    front: str
    back: str

@app.post('/aadhaar/ocr')
def execute_aadhaar_ocr(request: AadhaarOcrRequest):
    try:
        result = aadhaar_pipeline.run(
            base_aadhaar_pipieline_config,
            [request.front, request.back]
        )

        return result
    except Exception as e:
        return {
            'status': 'INTERNAL_SERVER_EXCEPTION',
            'message': str(e)
        }

# TODO: handle exceptions
# TODO: Send standard response
# TODO: get config from database, make them overridable from request body