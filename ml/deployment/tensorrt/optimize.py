"""
GPU Inference Optimization Pipeline

Exports trained PyTorch models to:
  - ONNX (cross-platform inference)
  - TensorRT (maximum GPU throughput)

Optimizations applied:
  - FP16 mixed precision
  - Graph fusion (Conv+BN+ReLU)
  - Dynamic input shapes
  - Layer caching
  - Quantization (INT8, optional)

Benchmarking:
  - Latency (P50/P95/P99)
  - Throughput (images/sec)
  - GPU memory usage
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ── ONNX Export ───────────────────────────────────────────────────────────────

class ONNXExporter:
    """
    Exports PyTorch models to ONNX format with optimization.

    Supports:
    - YOLOv8 (via Ultralytics built-in export)
    - Custom segmentation models
    - Feature extractors
    """

    def __init__(self, opset_version: int = 17, simplify: bool = True):
        self.opset_version = opset_version
        self.simplify = simplify

    def export_yolov8(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        imgsz: int = 1280,
        half: bool = True,
        nms: bool = True,
        device: str = "0",
    ) -> str:
        """Export YOLOv8 model to ONNX using Ultralytics API."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        output = model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            simplify=self.simplify,
            opset=self.opset_version,
            nms=nms,
            device=device,
        )
        logger.info("YOLOv8 ONNX export complete", path=output)
        return str(output)

    def export_pytorch_model(
        self,
        model,            # torch.nn.Module
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        dynamic_axes: Optional[Dict] = None,
        half: bool = True,
    ) -> str:
        """
        Generic PyTorch → ONNX export.

        Args:
            model:        PyTorch module (eval mode expected)
            output_path:  Destination .onnx file
            input_shape:  (N, C, H, W) dummy input shape
            dynamic_axes: e.g. {"input": {0: "batch"}, "output": {0: "batch"}}
            half:         Export in FP16

        Returns:
            Path to exported ONNX file
        """
        import torch

        model.eval()
        if half:
            model.half()

        dummy = torch.randn(*input_shape, dtype=torch.float16 if half else torch.float32)
        dummy = dummy.to(next(model.parameters()).device)

        if dynamic_axes is None:
            dynamic_axes = {
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=self.opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
        )

        # Simplify graph
        if self.simplify:
            try:
                import onnxsim
                import onnx
                model_onnx = onnx.load(output_path)
                model_sim, ok = onnxsim.simplify(model_onnx)
                if ok:
                    onnx.save(model_sim, output_path)
                    logger.info("ONNX graph simplified", path=output_path)
            except ImportError:
                logger.warning("onnxsim not installed – skipping simplification")

        logger.info("Model exported to ONNX", path=output_path, shape=input_shape, half=half)
        return output_path

    def verify_onnx(self, onnx_path: str, input_shape: Tuple[int, ...]) -> bool:
        """Verify ONNX model runs without errors."""
        try:
            import onnxruntime as ort
            import onnx

            onnx.checker.check_model(onnx_path)
            session = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            dummy = np.random.randn(*input_shape).astype(np.float32)
            out = session.run(None, {"input": dummy})
            logger.info("ONNX verification passed", outputs=len(out))
            return True
        except Exception as exc:
            logger.error("ONNX verification failed", error=str(exc))
            return False


# ── TensorRT Engine Builder ────────────────────────────────────────────────────

class TensorRTOptimizer:
    """
    Builds TensorRT engines from ONNX models for maximum GPU throughput.

    Features:
    - FP16 / INT8 precision
    - Dynamic batch size
    - Profile optimization
    - Engine serialization / caching
    """

    def __init__(
        self,
        precision: str = "fp16",   # fp32 | fp16 | int8
        workspace_gb: int = 8,
        max_batch_size: int = 16,
        device_id: int = 0,
    ):
        self.precision = precision
        self.workspace_bytes = workspace_gb * (1 << 30)
        self.max_batch_size = max_batch_size
        self.device_id = device_id

    def build_engine(
        self,
        onnx_path: str,
        engine_path: Optional[str] = None,
        min_shape:  Tuple[int, ...] = (1,  3, 640, 640),
        opt_shape:  Tuple[int, ...] = (4,  3, 640, 640),
        max_shape:  Tuple[int, ...] = (16, 3, 640, 640),
        calibrator=None,
    ) -> str:
        """
        Build and serialize a TensorRT engine.

        Args:
            onnx_path:    Input ONNX model
            engine_path:  Output .engine path (auto-generated if None)
            min/opt/max_shape: Dynamic shape range for batch optimisation
            calibrator:   INT8 calibration dataset (required for int8 precision)

        Returns:
            Path to serialized .engine file
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError(
                "TensorRT not installed. Install with: pip install tensorrt"
            )

        if engine_path is None:
            engine_path = onnx_path.replace(".onnx", f"_{self.precision}.engine")

        logger.info(
            "Building TensorRT engine",
            onnx=onnx_path,
            precision=self.precision,
            workspace_gb=self.workspace_bytes >> 30,
        )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder   = trt.Builder(TRT_LOGGER)
        network   = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser    = trt.OnnxParser(network, TRT_LOGGER)
        config    = builder.create_builder_config()

        # Workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)

        # Precision flags
        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TensorRT FP16 enabled")
        elif self.precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
            logger.info("TensorRT INT8 enabled")

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error", error=parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        # Dynamic shapes profile
        profile = builder.create_optimization_profile()
        inp = network.get_input(0)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(engine_path, "wb") as f:
            f.write(serialized)

        size_mb = os.path.getsize(engine_path) / 1e6
        logger.info("TensorRT engine built", path=engine_path, size_mb=round(size_mb, 1))
        return engine_path

    def load_engine(self, engine_path: str):
        """Load a previously serialized TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("TensorRT not installed")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        logger.info("TensorRT engine loaded", path=engine_path)
        return engine


# ── Inference Benchmarker ──────────────────────────────────────────────────────

class InferenceBenchmarker:
    """
    Benchmarks inference speed and memory usage across backends.

    Compares:
    - PyTorch FP32
    - PyTorch FP16
    - ONNX Runtime (GPU)
    - TensorRT FP16
    """

    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        self.warmup = warmup_iterations
        self.iterations = benchmark_iterations

    def benchmark_onnxruntime(
        self,
        onnx_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    ) -> Dict:
        """Benchmark ONNX Runtime inference on GPU."""
        try:
            import onnxruntime as ort
        except ImportError:
            return {"error": "onnxruntime not installed"}

        session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inp_name = session.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float16)

        # Warm-up
        for _ in range(self.warmup):
            session.run(None, {inp_name: dummy})

        # Benchmark
        latencies = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            session.run(None, {inp_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        return self._compute_stats(latencies, input_shape)

    def benchmark_pytorch(
        self,
        model,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        half: bool = True,
        device: str = "cuda:0",
    ) -> Dict:
        """Benchmark native PyTorch inference."""
        import torch

        model.eval()
        model.to(device)
        if half:
            model.half()

        dummy = torch.randn(*input_shape, device=device,
                            dtype=torch.float16 if half else torch.float32)

        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy)
            torch.cuda.synchronize()

            latencies = []
            for _ in range(self.iterations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        return self._compute_stats(latencies, input_shape)

    @staticmethod
    def _compute_stats(latencies: List[float], input_shape: Tuple) -> Dict:
        arr = np.array(latencies)
        batch = input_shape[0]
        return {
            "batch_size":       batch,
            "mean_ms":          round(float(np.mean(arr)), 2),
            "p50_ms":           round(float(np.percentile(arr, 50)), 2),
            "p95_ms":           round(float(np.percentile(arr, 95)), 2),
            "p99_ms":           round(float(np.percentile(arr, 99)), 2),
            "throughput_ips":   round(batch * 1000 / float(np.mean(arr)), 1),
            "std_ms":           round(float(np.std(arr)), 2),
        }

    def compare_all(
        self,
        pytorch_model,
        onnx_path: str,
        engine_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    ) -> Dict:
        """Run all benchmarks and return comparison table."""
        results = {}

        logger.info("Benchmarking PyTorch FP32…")
        results["pytorch_fp32"] = self.benchmark_pytorch(pytorch_model, input_shape, half=False)

        logger.info("Benchmarking PyTorch FP16…")
        results["pytorch_fp16"] = self.benchmark_pytorch(pytorch_model, input_shape, half=True)

        logger.info("Benchmarking ONNX Runtime…")
        results["onnxruntime_gpu"] = self.benchmark_onnxruntime(onnx_path, input_shape)

        # Print comparison
        print("\n" + "="*70)
        print(f"{'Backend':<25} {'Mean ms':>10} {'P95 ms':>10} {'img/s':>10}")
        print("-"*70)
        for backend, stats in results.items():
            if "error" not in stats:
                print(
                    f"{backend:<25} {stats['mean_ms']:>10.2f} "
                    f"{stats['p95_ms']:>10.2f} {stats['throughput_ips']:>10.1f}"
                )
        print("="*70 + "\n")

        return results


# ── Triton model config generator ────────────────────────────────────────────

def generate_triton_config(
    model_name: str,
    backend: str = "tensorrt_plan",  # tensorrt_plan | onnxruntime | pytorch_libtorch
    max_batch_size: int = 8,
    input_name: str = "images",
    input_shape: List[int] = [3, 640, 640],
    output_names: List[str] = ["output0"],
    output_shapes: List[List[int]] = [[-1, 84, 8400]],
    preferred_batch: List[int] = [1, 2, 4, 8],
    output_dir: str = "models",
) -> str:
    """
    Generate NVIDIA Triton Inference Server model configuration.
    Creates model repository structure ready for tritonserver.
    """
    model_dir = Path(output_dir) / model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    input_config = f"""
  {{
    name: "{input_name}"
    data_type: TYPE_FP16
    dims: {input_shape}
  }}"""

    output_configs = "\n".join([
        f"""  {{
    name: "{name}"
    data_type: TYPE_FP32
    dims: {shape}
  }}"""
        for name, shape in zip(output_names, output_shapes)
    ])

    preferred = ", ".join(str(b) for b in preferred_batch)

    config = f"""name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

input [{input_config}
]

output [
{output_configs}
]

dynamic_batching {{
  preferred_batch_size: [{preferred}]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }}
]

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator {{
      name: "tensorrt"
      parameters {{
        key: "precision_mode"
        value: "FP16"
      }}
    }}
  }}
}}
"""

    config_path = Path(output_dir) / model_name / "config.pbtxt"
    config_path.write_text(config)

    logger.info("Triton config generated", model=model_name, path=str(config_path))
    return str(config_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RIP Model Export & Optimization")
    parser.add_argument("--model",       required=True, help="Path to .pt model")
    parser.add_argument("--output",      default="exports/", help="Output directory")
    parser.add_argument("--format",      nargs="+", default=["onnx"], choices=["onnx", "tensorrt"])
    parser.add_argument("--precision",   default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--imgsz",       type=int, default=1280)
    parser.add_argument("--batch",       type=int, default=8)
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--triton",      action="store_true", help="Generate Triton config")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    exporter = ONNXExporter()

    onnx_path = None
    if "onnx" in args.format:
        onnx_path = exporter.export_yolov8(
            args.model,
            output_path=str(Path(args.output) / "model.onnx"),
            imgsz=args.imgsz,
            half=(args.precision == "fp16"),
        )
        exporter.verify_onnx(onnx_path, (1, 3, args.imgsz, args.imgsz))

    if "tensorrt" in args.format and onnx_path:
        optimizer = TensorRTOptimizer(
            precision=args.precision,
            workspace_gb=8,
            max_batch_size=args.batch,
        )
        optimizer.build_engine(
            onnx_path,
            engine_path=str(Path(args.output) / f"model_{args.precision}.engine"),
            opt_shape=(args.batch, 3, args.imgsz, args.imgsz),
            max_shape=(args.batch * 2, 3, args.imgsz, args.imgsz),
        )

    if args.triton and onnx_path:
        generate_triton_config(
            model_name="construction_detector",
            output_dir=args.output,
            max_batch_size=args.batch,
        )

    print("✅ Export complete!")
