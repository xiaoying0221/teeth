from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    saved_path = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    detections = relationship("DetectionRecord", back_populates="image", cascade="all, delete-orphan")


class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    class_id = Column(Integer, default=0, nullable=False)
    class_name = Column(String, default="abnormal", nullable=False)
    confidence = Column(Float, nullable=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    created_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    image = relationship("ImageRecord", back_populates="detections")
    corrections = relationship("CorrectionRecord", back_populates="detection", cascade="all, delete-orphan")


class CorrectionRecord(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False, index=True)
    box_index = Column(Integer, default=0, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    original_area = Column(Float, nullable=False)
    edited_area = Column(Float, nullable=False)
    intersection_area = Column(Float, nullable=False)
    union_area = Column(Float, nullable=False)
    iou = Column(Float, nullable=False)
    difference_ratio = Column(Float, nullable=False)
    corrected_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    detection = relationship("DetectionRecord", back_populates="corrections")
