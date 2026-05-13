# Kinematic Engine | Real-Time Optical Fluid Dynamics

Kinematic Engine es un motor de visión computacional y renderizado híbrido que simula físicas de fluidos ópticos y refracción en tiempo real. 

El sistema optimiza el flujo de datos mediante un pipeline que combina **Computer Vision tradicional (OpenCV)** para la ingesta y pre-procesamiento, con **cálculo tensorial masivo en GPU (PyTorch CUDA)** para la simulación física y el remapeo óptico.

![Kinematic Engine Demo](./assets/demo.gif)

## 🚀 Arquitectura Técnica

El motor implementa una arquitectura desacoplada para maximizar los FPS (Frames Per Second) incluso bajo cargas pesadas de inferencia:

1.  **Threaded Ingestion:** La captura de vídeo se ejecuta en un hilo asíncrono para eliminar la latencia de red (especialmente útil al usar webcams inalámbricas/móviles).
2.  **Dense Optical Flow (CPU):** Se utiliza el algoritmo de Farneback optimizado geométricamente a resolución reducida ($0.5x$) para extraer vectores de inercia sin saturar el procesador.
3.  **Inference Hot-Swapping:**
    * **2D Mode:** Segmentación semántica (Selfie/FaceMesh) para aislamiento de masas.
    * **3D Mode (MiDaS):** Estimación de profundidad monocular. Implementa una función de activación exponencial ($Z^{3.5}$) para simular colisiones físicas frontales e inercia volumétrica.
4.  **Tensor Graphics (GPU):** Los datos se inyectan en el bus PCIe hacia la VRAM. PyTorch gestiona la memoria térmica (viscosidad) y ejecuta un `F.grid_sample` bilineal para la deformación de la matriz de vídeo por hardware.

## 🛠️ Solución de Desafíos de Ingeniería

* **Memory Contiguity:** Se resolvió la incompatibilidad de punteros entre tensores de PyTorch y matrices de OpenCV mediante la reestructuración de la RAM con `np.ascontiguousarray`, evitando errores de layout en el Frame Buffer de C++.
* **Blackwell Architecture Support:** Configuración de compatibilidad para GPUs NVIDIA Serie 5000 (Compute Capability `sm_120`) mediante el uso de *Nightly Builds* y controladores CUDA 12.x.

## 📦 Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/TuUsuario/kinematic-engine.git
    cd kinematic-engine
    ```

2.  **Instalar dependencias base:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Instalar PyTorch (Versión recomendada para GPU):**
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    ```

4.  **Ejecutar:**
    ```bash
    python src/kinematic_engine.py
    ```

## 📝 Licencia
Este proyecto se distribuye bajo la licencia MIT.
