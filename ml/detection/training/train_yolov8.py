"""
YOLOv8 Construction Object Detection - Training Pipeline

Trains YOLOv8 to detect construction-specific objects:
- Workers (with/without PPE)
- Heavy equipment (cranes, excavators, concrete mixers, bulldozers)
- Structural elements (columns, slabs, beams, rebar, walls)
- Materials (concrete bags, steel coils, lumber, pipes)
- Scaffolding and formwork
- Safety equipment (helmets, vests, barriers)
- Vehicles (trucks, forklifts)

Uses:
- Ultralytics YOLOv8
- MLflow experiment tracking
- Weights & Biases integration
- Mixed precision training (FP16)
- Multi-GPU DDP training
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import mlflow
import mlflow.pytorch
import structlog
from ultralytics import YOLO
from ultralytics.utils.callbacks.mlflow import on_fit_epoch_end

logger = structlog.get_logger(__name__)


# =============================================================================
# CONSTRUCTION DETECTION CLASSES
# =============================================================================

CONSTRUCTION_CLASSES = {
    # Workers
    0: "worker_with_ppe",
    1: "worker_without_ppe",
    2: "worker_partial_ppe",

    # Heavy equipment
    3: "tower_crane",
    4: "mobile_crane",
    5: "excavator",
    6: "concrete_mixer",
    7: "bulldozer",
    8: "compactor",
    9: "forklift",
    10: "aerial_work_platform",

    # Structural elements
    11: "column_concrete",
    12: "column_steel",
    13: "slab_concrete",
    14: "beam_concrete",
    15: "beam_steel",
    16: "rebar_bundle",
    17: "rebar_installed",
    18: "wall_concrete",
    19: "wall_masonry",
    20: "foundation",

    # Temporary structures
    21: "scaffolding",
    22: "formwork",
    23: "construction_barrier",

    # Materials
    24: "concrete_bags",
    25: "steel_coils",
    26: "lumber_stack",
    27: "pipe_bundle",
    28: "brick_stack",

    # Safety
    29: "safety_cone",
    30: "safety_net",
    31: "hard_hat",
    32: "safety_vest",

    # Vehicles
    33: "concrete_truck",
    34: "dump_truck",
    35: "pickup_truck",
}

NUM_CLASSES = len(CONSTRUCTION_CLASSES)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

DEFAULT_TRAIN_CONFIG = {
    # Model
    "model": "yolov8l.pt",  # yolov8n/s/m/l/x
    "task": "detect",

    # Data
    "data": "configs/construction_detection.yaml",
    "imgsz": 1280,  # High-res for construction sites

    # Training
    "epochs": 300,
    "patience": 50,
    "batch": 16,
    "workers": 8,
    "cache": True,  # Cache images in RAM

    # Optimization
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "warmup_momentum": 0.8,

    # Augmentation (heavy for construction diversity)
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,        # Rotation (drones can have various angles)
    "translate": 0.1,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0003,
    "flipud": 0.05,         # Occasional vertical flip for drone footage
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,      # Good for small objects (workers, rebar)
    "auto_augment": "randaugment",

    # Loss
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # Precision
    "half": True,           # FP16 training

    # Output
    "project": "outputs/detection",
    "name": f"construction_yolov8l_{datetime.now().strftime('%Y%m%d_%H%M')}",
    "save": True,
    "save_period": 10,
    "plots": True,
    "val": True,
}


# =============================================================================
# DATASET CONFIGURATION GENERATOR
# =============================================================================

def generate_dataset_config(
    dataset_dir: str,
    output_path: str = "configs/construction_detection.yaml",
) -> str:
    """
    Generate YAML dataset configuration for YOLOv8.
    Expects COCO-format annotations converted to YOLO format.
    """
    dataset_dir = Path(dataset_dir)

    config = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": list(CONSTRUCTION_CLASSES.values()),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Dataset config generated", path=output_path)
    return str(output_path)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class YOLOv8ConstructionTrainer:
    """
    Production-grade YOLOv8 training pipeline for construction detection.

    Features:
    - MLflow experiment tracking
    - Weights & Biases integration
    - Automatic checkpoint management
    - Multi-GPU support
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mlflow_uri: str = "http://localhost:5000",
        wandb_project: str = "rip-construction-detection",
        experiment_name: str = "yolov8-construction",
    ):
        self.config = {**DEFAULT_TRAIN_CONFIG, **(config or {})}
        self.mlflow_uri = mlflow_uri
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name

        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        # Setup W&B
        if os.getenv("WANDB_API_KEY"):
            import wandb
            wandb.init(
                project=wandb_project,
                config=self.config,
                tags=["yolov8", "construction", "detection"],
            )

        logger.info(
            "Trainer initialized",
            model=self.config["model"],
            num_classes=NUM_CLASSES,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def train(
        self,
        dataset_path: str,
        resume: bool = False,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.

        Args:
            dataset_path: Path to dataset directory
            resume: Resume from last checkpoint
            device: Device string (e.g., "0", "0,1", "cpu")

        Returns:
            Training results dictionary
        """
        # Generate dataset config
        data_config = generate_dataset_config(dataset_path)
        self.config["data"] = data_config

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = ",".join(str(i) for i in range(torch.cuda.device_count()))
                logger.info("Using GPUs", devices=device, count=torch.cuda.device_count())
            else:
                device = "cpu"
                logger.warning("No GPU available, training on CPU (will be slow)")

        self.config["device"] = device

        with mlflow.start_run(run_name=self.config["name"]) as run:
            # Log parameters
            mlflow.log_params(self.config)
            mlflow.log_param("num_classes", NUM_CLASSES)
            mlflow.log_param("dataset_path", dataset_path)

            logger.info(
                "Starting training",
                run_id=run.info.run_id,
                config=self.config,
            )

            # Initialize model
            if resume:
                # Resume from last checkpoint
                checkpoint_path = Path(self.config["project"]) / self.config["name"] / "weights" / "last.pt"
                if checkpoint_path.exists():
                    model = YOLO(str(checkpoint_path))
                    logger.info("Resuming from checkpoint", path=checkpoint_path)
                else:
                    logger.warning("Checkpoint not found, starting fresh")
                    model = YOLO(self.config["model"])
            else:
                model = YOLO(self.config["model"])

            # Log model architecture info
            mlflow.log_param("model_parameters", sum(p.numel() for p in model.model.parameters()))

            # Train
            train_args = {k: v for k, v in self.config.items() if k not in ["model"]}
            results = model.train(**train_args)

            # Evaluate
            metrics = model.val(data=data_config, device=device, half=True)

            # Log metrics
            final_metrics = {
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }

            # Per-class AP
            for class_id, class_name in CONSTRUCTION_CLASSES.items():
                if class_id < len(metrics.box.ap_class_index):
                    ap = float(metrics.box.ap_class_index[class_id]) if class_id in metrics.box.ap_class_index else 0.0
                    final_metrics[f"AP50_{class_name}"] = ap

            mlflow.log_metrics(final_metrics)

            # Log model artifact
            best_model_path = (
                Path(self.config["project"])
                / self.config["name"]
                / "weights"
                / "best.pt"
            )
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), "model_weights")
                mlflow.pytorch.log_model(model.model, "pytorch_model")

            logger.info("Training completed", metrics=final_metrics)

            return {
                "run_id": run.info.run_id,
                "model_path": str(best_model_path),
                "metrics": final_metrics,
            }

    def export(
        self,
        model_path: str,
        formats: list = ["onnx", "engine"],  # ONNX + TensorRT
        device: str = "0",
        half: bool = True,
    ) -> Dict[str, str]:
        """
        Export trained model to multiple formats.

        Formats:
        - onnx: ONNX Runtime inference
        - engine: TensorRT engine (fastest GPU inference)
        - coreml: Apple CoreML (mobile)
        - tflite: TensorFlow Lite (edge)
        """
        model = YOLO(model_path)
        exported_paths = {}

        for fmt in formats:
            logger.info("Exporting model", format=fmt, half=half)
            try:
                export_path = model.export(
                    format=fmt,
                    half=half,
                    device=device,
                    simplify=True,  # Simplify ONNX graph
                    opset=17,
                    workspace=8,  # TensorRT workspace GB
                    nms=True,     # Include NMS in export
                )
                exported_paths[fmt] = str(export_path)
                logger.info("Export successful", format=fmt, path=export_path)
            except Exception as e:
                logger.error("Export failed", format=fmt, error=str(e))

        return exported_paths


# =============================================================================
# INFERENCE CLASS (for production use)
# =============================================================================

class YOLOv8ConstructionDetector:
    """
    Production-grade inference wrapper for YOLOv8.

    Features:
    - Batch inference
    - GPU memory management
    - Mixed precision (FP16)
    - ONNX Runtime support
    - TensorRT acceleration
    - Confidence + NMS thresholding
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_half: bool = True,
        use_onnx: bool = False,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = CONSTRUCTION_CLASSES

        logger.info("Loading detection model", path=model_path, device=device)

        if use_onnx:
            self._load_onnx(model_path)
        else:
            self.model = YOLO(model_path)
            if use_half and "cuda" in device:
                self.model.model.half()

        self.model.to(device)
        logger.info("Detection model loaded", num_classes=NUM_CLASSES)

    def _load_onnx(self, model_path: str):
        """Load ONNX model with ONNX Runtime."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        self.use_onnx = True
        logger.info("ONNX model loaded", providers=providers)

    def detect(
        self,
        images,  # np.ndarray or List[np.ndarray]
        batch_size: int = 8,
    ) -> list:
        """
        Run inference on images.

        Args:
            images: Single image or list of images (HWC BGR)
            batch_size: Batch size for GPU inference

        Returns:
            List of detection results, each containing:
            {
                "detections": [
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2],
                        "bbox_normalized": [x1n, y1n, x2n, y2n],
                    }
                ],
                "count_by_class": {"worker_with_ppe": 3, ...},
                "inference_time_ms": float,
            }
        """
        import time
        import numpy as np

        if not isinstance(images, list):
            images = [images]

        all_results = []

        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            t0 = time.perf_counter()

            results = self.model(
                batch,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                stream=False,
            )

            inference_time_ms = (time.perf_counter() - t0) * 1000 / len(batch)

            for result in results:
                img_h, img_w = result.orig_shape

                detections = []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "bbox_normalized": [
                            float(x1 / img_w), float(y1 / img_h),
                            float(x2 / img_w), float(y2 / img_h),
                        ],
                    })

                # Aggregate counts
                count_by_class = {}
                for det in detections:
                    cn = det["class_name"]
                    count_by_class[cn] = count_by_class.get(cn, 0) + 1

                all_results.append({
                    "detections": detections,
                    "count_by_class": count_by_class,
                    "total_detections": len(detections),
                    "inference_time_ms": round(inference_time_ms, 2),
                })

        return all_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Construction Detection Training")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--model", default="yolov8l.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default=None, help="GPU device(s)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", action="store_true", help="Export after training")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--wandb-project", default="rip-construction-detection")
    parser.add_argument("--config", default=None, help="Path to custom config YAML")

    args = parser.parse_args()

    # Load custom config if provided
    custom_config = {}
    if args.config:
        with open(args.config) as f:
            custom_config = yaml.safe_load(f)

    # Merge with CLI args
    config = {
        **custom_config,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.image_size,
    }

    trainer = YOLOv8ConstructionTrainer(
        config=config,
        mlflow_uri=args.mlflow_uri,
        wandb_project=args.wandb_project,
    )

    result = trainer.train(
        dataset_path=args.dataset,
        resume=args.resume,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model: {result['model_path']}")
    print(f"mAP50: {result['metrics']['mAP50']:.4f}")
    print(f"mAP50-95: {result['metrics']['mAP50_95']:.4f}")
    print(f"{'='*60}\n")

    if args.export:
        exported = trainer.export(result["model_path"])
        print("Exported models:")
        for fmt, path in exported.items():
            print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
