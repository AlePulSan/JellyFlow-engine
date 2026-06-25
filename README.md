# JellyFlow Engine

A hybrid computer vision engine built to simulate optical fluid physics and real-time refraction. 

The core design goal was to avoid CPU bottlenecks. To achieve high frame rates with concurrent inference models running, the pipeline strictly decouples OpenCV's CPU processing from the heavy tensor math handled by the GPU (PyTorch/CUDA).

![System Demo](./assets/JellyFlow.gif)

## Technical Architecture

I built the pipeline around an asynchronous data flow to minimize ingestion and processing latency:

* **Async Ingestion (Anti-Lag):** Video capture runs on a dedicated background thread. This absorbs network latency when pulling feeds from IP cameras or wireless mobile devices, keeping the main render loop completely unblocked.
* **Vector Field (Optical Flow):** Uses Farnebäck's algorithm to extract inertia vectors. I run this at half resolution (0.5x) to save CPU cycles before offloading the interpolation to the GPU.
* **Hot-Swappable Inference:**
    * **2D Mode (Semantic):** Uses MediaPipe for body and face segmentation to isolate the kinetic masses.
    * **3D Mode (Depth):** Monocular depth estimation via MiDaS. I implemented a custom activation threshold and an exponential scale (Z^3.5) to simulate frontal physical collisions based on target proximity.
* **GPU Tensor Math:** Flow vectors and inference masks are pushed directly to VRAM. PyTorch handles the "thermal" memory (viscosity) and runs a bilinear sampling function (`F.grid_sample`) to geometrically warp the frame buffer straight from the hardware.

## Memory Management & Performance

* **Memory Contiguity:** Passing pointers back and forth between PyTorch tensors and OpenCV arrays usually breaks memory layouts. The engine fixes this RAM layout on the fly using `np.ascontiguousarray`, guaranteeing data integrity when pulling frames back from GPU space.
* **Hardware Tuning:** Compiled and optimized specifically for high-performance NVIDIA architectures running the latest CUDA builds.

## Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/AlePulSan/JellyFlow-engine.git](https://github.com/AlePulSan/JellyFlow-engine.git)
cd JellyFlow-engine
```

2. **Install base dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA support:**
```bash
pip install --pre torch torchvision --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)
```

## Usage

To run the engine with the default configuration:

```bash
python src/JellyFlow_engine.py
```

## Input Source Configuration

The system automatically detects the available camera, but it can be configured to process local video files. Modify the constant in `src/JellyFlow_engine.py`:

```python
# For live camera:
VIDEO_SOURCE = 0

# To process a pre-recorded file:
VIDEO_SOURCE = "path/to/video.mp4"
```

The engine standardizes the input resolution to 720p and handles the playback loop automatically.


## Sandbox (Proof of Concepts)

To keep the core engine stable and modular, individual subsystems are isolated, stress-tested, and validated in the `pocs/` directory. This acts as a testing and learning playground before integrating code into the main pipeline:

* **`camera/` (I/O Isolation):** Multithreaded Producer-Consumer pattern to solve buffer bloat and prevent camera lag.
* **`tracker/` (CPU Telemetry):** Pure spatial landmark extraction and convex hull calculations using MediaPipe.
* **`gpu/` (Memory Management):** Isolated PyTorch distortion engine directly manipulating tensor memory on the GPU.
