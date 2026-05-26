from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from ultralytics import YOLO

from .database import get_db, init_db
from .models import AddedBoxRecord, CorrectionRecord, DeletedBoxRecord, DetectionRecord, ImageRecord

BEST_PT = Path(r"d:\文档\毕业设计\teeth\runs\exp1_yolov8n\weights\best.pt")
MODEL_NAME = "exp6_yolov8s_cbam_transfer"
BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Teeth Detection API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

model: Optional[YOLO] = None


class BoxPayload(BaseModel):
    x: float
    y: float
    width: float
    height: float


class CompareRequest(BaseModel):
    original_box: BoxPayload
    edited_box: BoxPayload
    detection_id: Optional[int] = None
    box_index: Optional[int] = 0


class AddedBoxCreateRequest(BaseModel):
    image_id: int
    box_index: Optional[int] = 0
    box: BoxPayload


class AddedBoxUpdateRequest(BaseModel):
    box: BoxPayload


class DeleteBoxRequest(BaseModel):
    image_id: int
    detection_id: int
    box_index: int


def calc_compare(original: BoxPayload, edited: BoxPayload):
    area1 = max(0.0, original.width) * max(0.0, original.height)
    area2 = max(0.0, edited.width) * max(0.0, edited.height)

    x1 = max(original.x, edited.x)
    y1 = max(original.y, edited.y)
    x2 = min(original.x + original.width, edited.x + edited.width)
    y2 = min(original.y + original.height, edited.y + edited.height)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection_area = inter_w * inter_h
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    difference_ratio = 1 - iou

    return {
        "original_area": round(area1, 2),
        "edited_area": round(area2, 2),
        "intersection_area": round(intersection_area, 2),
        "union_area": round(union_area, 2),
        "iou": round(iou, 4),
        "difference_ratio": round(difference_ratio, 4),
    }


def image_to_dict(record: ImageRecord, added_boxes=None, deleted_box_indexes=None):
    detections = list(record.detections or [])
    all_corrections = []
    for det in detections:
        all_corrections.extend(det.corrections or [])
    first_detection = detections[0] if detections else None

    image_url = f"/uploads/{Path(record.saved_path).name}" if record.saved_path else None
    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "image_url": image_url,
        "url": image_url,
        "width": record.width,
        "height": record.height,
        "upload_time": record.upload_time.isoformat(),
        "detections": [detection_to_dict(item) for item in detections],
        "detection": detection_to_dict(first_detection) if first_detection else None,
        "corrections": [correction_to_dict(item) for item in all_corrections],
        "added_boxes": [added_box_to_dict(item) for item in (added_boxes or [])],
        "deleted_box_indexes": sorted(list(set(deleted_box_indexes or []))),
        "latest_correction": correction_to_dict(all_corrections[-1]) if all_corrections else None,
    }


def detection_to_dict(record: Optional[DetectionRecord]):
    if record is None:
        return None
    return {
        "id": record.id,
        "image_id": record.image_id,
        "model_name": record.model_name,
        "bbox": {
            "x": record.x,
            "y": record.y,
            "width": record.width,
            "height": record.height,
        },
        "confidence": record.confidence,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "created_time": record.created_time.isoformat(),
    }


def correction_to_dict(record: Optional[CorrectionRecord]):
    if record is None:
        return None
    return {
        "id": record.id,
        "detection_id": record.detection_id,
        "bbox": {
            "x": record.x,
            "y": record.y,
            "width": record.width,
            "height": record.height,
        },
        "original_area": record.original_area,
        "edited_area": record.edited_area,
        "intersection_area": record.intersection_area,
        "union_area": record.union_area,
        "iou": record.iou,
        "difference_ratio": record.difference_ratio,
        "box_index": getattr(record, "box_index", 0),
        "corrected_time": record.corrected_time.isoformat(),
    }


def added_box_to_dict(record: Optional[AddedBoxRecord]):
    if record is None:
        return None
    return {
        "id": record.id,
        "image_id": record.image_id,
        "box_index": record.box_index,
        "bbox": {
            "x": record.x,
            "y": record.y,
            "width": record.width,
            "height": record.height,
        },
        "created_time": record.created_time.isoformat(),
    }


@app.on_event("startup")
def startup():
    global model
    init_db()
    if not BEST_PT.exists():
        raise RuntimeError(f"模型权重不存在: {BEST_PT}")
    model = YOLO(str(BEST_PT))


@app.get("/")
def root():
    return {"message": "Teeth Detection API running", "model": str(BEST_PT)}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片读取失败: {exc}") from exc

    width, height = image.size
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    safe_name = f"{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / safe_name
    image.save(saved_path)

    image_record = ImageRecord(
        filename=file.filename or safe_name,
        saved_path=str(saved_path),
        width=width,
        height=height,
    )
    db.add(image_record)
    db.commit()
    db.refresh(image_record)

    results = model.predict(source=image, conf=0.25, verbose=False, device="0")
    result = results[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()

        for bbox, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = bbox
            box = {
                "x": round(x1, 2),
                "y": round(y1, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            }
            detection_record = DetectionRecord(
                image_id=image_record.id,
                model_name=MODEL_NAME,
                class_id=int(cls_id),
                class_name="abnormal",
                confidence=round(conf, 4),
                x=box["x"],
                y=box["y"],
                width=box["width"],
                height=box["height"],
            )
            db.add(detection_record)
            db.commit()
            db.refresh(detection_record)
            detections.append(
                {
                    "id": detection_record.id,
                    "bbox": box,
                    "confidence": round(conf, 4),
                    "class_id": int(cls_id),
                    "class_name": "abnormal",
                }
            )

    image_url = f"/uploads/{safe_name}"
    return {
        "success": True,
        "image_id": image_record.id,
        "image": {
            "id": image_record.id,
            "width": width,
            "height": height,
            "name": file.filename,
            "url": image_url,
            "image_url": image_url,
        },
        "detections": detections,
    }


@app.post("/api/compare")
def compare(payload: CompareRequest, db: Session = Depends(get_db)):
    result = calc_compare(payload.original_box, payload.edited_box)
    correction_id = None

    if payload.detection_id is not None:
        detection = db.get(DetectionRecord, payload.detection_id)
        if detection is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")

        correction = CorrectionRecord(
            detection_id=payload.detection_id,
            box_index=payload.box_index or 0,
            x=round(payload.edited_box.x, 2),
            y=round(payload.edited_box.y, 2),
            width=round(payload.edited_box.width, 2),
            height=round(payload.edited_box.height, 2),
            original_area=result["original_area"],
            edited_area=result["edited_area"],
            intersection_area=result["intersection_area"],
            union_area=result["union_area"],
            iou=result["iou"],
            difference_ratio=result["difference_ratio"],
        )
        db.add(correction)
        db.commit()
        db.refresh(correction)
        correction_id = correction.id

    return {"success": True, "correction_id": correction_id, "box_index": payload.box_index, **result}


@app.post("/api/added-boxes")
def create_added_box(payload: AddedBoxCreateRequest, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="图片记录不存在")

    added_box = AddedBoxRecord(
        image_id=payload.image_id,
        box_index=payload.box_index or 0,
        x=round(payload.box.x, 2),
        y=round(payload.box.y, 2),
        width=round(payload.box.width, 2),
        height=round(payload.box.height, 2),
    )
    db.add(added_box)
    db.commit()
    db.refresh(added_box)
    return {"success": True, "added_box": added_box_to_dict(added_box)}


@app.put("/api/added-boxes/{added_box_id}")
def update_added_box(added_box_id: int, payload: AddedBoxUpdateRequest, db: Session = Depends(get_db)):
    added_box = db.get(AddedBoxRecord, added_box_id)
    if added_box is None:
        raise HTTPException(status_code=404, detail="新增修正框不存在")

    added_box.x = round(payload.box.x, 2)
    added_box.y = round(payload.box.y, 2)
    added_box.width = round(payload.box.width, 2)
    added_box.height = round(payload.box.height, 2)
    db.commit()
    db.refresh(added_box)
    return {"success": True, "added_box": added_box_to_dict(added_box)}


@app.post("/api/deleted-boxes")
def create_deleted_box(payload: DeleteBoxRequest, db: Session = Depends(get_db)):
    image = db.get(ImageRecord, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="图片记录不存在")

    detection = db.get(DetectionRecord, payload.detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="检测记录不存在")

    exists = (
        db.query(DeletedBoxRecord)
        .filter(
            DeletedBoxRecord.image_id == payload.image_id,
            DeletedBoxRecord.detection_id == payload.detection_id,
            DeletedBoxRecord.box_index == payload.box_index,
        )
        .first()
    )
    if exists is not None:
        return {"success": True, "deleted_box_id": exists.id, "box_index": payload.box_index}

    deleted_box = DeletedBoxRecord(
        image_id=payload.image_id,
        detection_id=payload.detection_id,
        box_index=payload.box_index,
    )
    db.add(deleted_box)
    db.commit()
    db.refresh(deleted_box)
    return {"success": True, "deleted_box_id": deleted_box.id, "box_index": payload.box_index}


@app.get("/api/records")
def list_records(db: Session = Depends(get_db)):
    records = (
        db.query(ImageRecord)
        .options(joinedload(ImageRecord.detections).joinedload(DetectionRecord.corrections))
        .order_by(ImageRecord.upload_time.desc())
        .limit(50)
        .all()
    )
    record_ids = [item.id for item in records]
    added_map = {}
    deleted_map = {}
    if record_ids:
        added_items = (
            db.query(AddedBoxRecord)
            .filter(AddedBoxRecord.image_id.in_(record_ids))
            .order_by(AddedBoxRecord.created_time.asc())
            .all()
        )
        for item in added_items:
            added_map.setdefault(item.image_id, []).append(item)

        deleted_items = (
            db.query(DeletedBoxRecord)
            .filter(DeletedBoxRecord.image_id.in_(record_ids))
            .all()
        )
        for item in deleted_items:
            deleted_map.setdefault(item.image_id, set()).add(item.box_index)

    return {
        "success": True,
        "records": [image_to_dict(item, added_map.get(item.id, []), deleted_map.get(item.id, set())) for item in records],
    }


@app.get("/api/records/{image_id}")
def get_record(image_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(ImageRecord)
        .options(joinedload(ImageRecord.detections).joinedload(DetectionRecord.corrections))
        .filter(ImageRecord.id == image_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    added_items = (
        db.query(AddedBoxRecord)
        .filter(AddedBoxRecord.image_id == image_id)
        .order_by(AddedBoxRecord.created_time.asc())
        .all()
    )
    deleted_indexes = {
        item.box_index
        for item in db.query(DeletedBoxRecord).filter(DeletedBoxRecord.image_id == image_id).all()
    }
    return {"success": True, "record": image_to_dict(record, added_items, deleted_indexes)}
