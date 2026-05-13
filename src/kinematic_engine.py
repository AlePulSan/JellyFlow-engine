import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import mediapipe as mp
import time
import warnings
from threading import Thread

warnings.filterwarnings("ignore")

# ==========================================
# 1. CLASE DE INGESTA ASÍNCRONA (ANTI-LAG)
# ==========================================
class ThreadedCamera:
    """Aisla la latencia de la cámara (WiFi/USB) en un hilo independiente."""
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened() and src == 0:
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened(): break
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# 2. CONFIGURACIÓN GLOBAL
# ==========================================
VIDEO_SOURCE = 0 # Cambia por "ruta/video.mp4" para procesar archivos
W_CAM, H_CAM = 1280, 720
UI_WIDTH = 300
CURRENT_MODE = "PROFUNDIDAD"

# Coordenadas de botones (Alineación corregida para el Mouse Callback)
BTN_X1 = W_CAM - UI_WIDTH + 30
BTN_X2 = W_CAM - 30
BTN_Y_CUERPO = (160, 200)
BTN_Y_CARAS = (220, 260)
BTN_Y_PROF = (280, 320)

# Físicas de profundidad (Efecto "Puñetazo")
Z_THRESHOLD = 0.82   
Z_EXP = 3.5          

def mouse_callback(event, x, y, flags, param):
    global CURRENT_MODE
    if event == cv2.EVENT_LBUTTONDOWN:
        # Detectamos el clic basándonos en los rectángulos dibujados
        if BTN_X1 <= x <= BTN_X2:
            if BTN_Y_CUERPO[0] <= y <= BTN_Y_CUERPO[1]: CURRENT_MODE = "CUERPO"
            elif BTN_Y_CARAS[0] <= y <= BTN_Y_CARAS[1]: CURRENT_MODE = "CARAS"
            elif BTN_Y_PROF[0] <= y <= BTN_Y_PROF[1]: CURRENT_MODE = "PROFUNDIDAD"

# ==========================================
# 3. INTERFAZ MINIMALISTA (HUD)
# ==========================================
def draw_hud(frame, fps, backend_str):
    # Panel lateral translúcido (70% oscuridad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (W_CAM - UI_WIDTH, 0), (W_CAM, H_CAM), (10, 10, 12), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame) 

    # Branding
    cv2.putText(frame, "KINEMATIC ENGINE", (BTN_X1, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"System: {backend_str}", (BTN_X1, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, f"Frame rate: {int(fps)} FPS", (BTN_X1, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.line(frame, (BTN_X1, 130), (BTN_X2, 130), (40, 40, 42), 1)

    # Botones Flat
    modes = [("CUERPO", BTN_Y_CUERPO), ("CARAS", BTN_Y_CARAS), ("PROFUNDIDAD", BTN_Y_PROF)]
    for text, (y1, y2) in modes:
        active = (CURRENT_MODE == text)
        bg = (245, 245, 245) if active else (22, 22, 25)
        txt = (10, 10, 10) if active else (130, 130, 130)
        
        cv2.rectangle(frame, (BTN_X1, y1), (BTN_X2, y2), bg, -1)
        cv2.putText(frame, text, (BTN_X1 + 20, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt, 1)

    # Telemetría de Profundidad (Sin solapamientos)
    if CURRENT_MODE == "PROFUNDIDAD":
        cv2.line(frame, (BTN_X1, 350), (BTN_X2, 350), (40, 40, 42), 1)
        cv2.putText(frame, "Z-DEPTH METRICS", (BTN_X1, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"Wall threshold: {Z_THRESHOLD}", (BTN_X1, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
        cv2.putText(frame, f"Impact factor: ^{Z_EXP}", (BTN_X1, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

# ==========================================
# 4. LOOP DE PRODUCCIÓN
# ==========================================
def main():
    print("[System] Inicializando Kinematic Engine...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modelos
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    # Cámara asíncrona
    cam = ThreadedCamera(VIDEO_SOURCE, W_CAM, H_CAM).start()
    
    cv2.namedWindow("Kinematic Engine", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Kinematic Engine", mouse_callback)

    # Memoria VRAM
    y, x = torch.meshgrid(torch.linspace(-1, 1, H_CAM, device=device), torch.linspace(-1, 1, W_CAM, device=device), indexing='ij')
    base_grid = torch.stack((x, y), dim=2).unsqueeze(0)
    flow_accumulator = torch.zeros((1, H_CAM, W_CAM, 2), device=device)

    prev_gray = None
    p_time = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None: continue
            
            frame = cv2.resize(frame, (W_CAM, H_CAM))
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5, fy=0.5)

            if prev_gray is None:
                prev_gray = curr_gray
                continue

            # 1. Optical Flow (CPU optimizada)
            flow_small = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.resize(flow_small, (W_CAM, H_CAM)) * 2.0
            prev_gray = curr_gray

            # 2. Selección de Máscara (IA)
            mask = np.zeros((H_CAM, W_CAM), dtype=np.float32)
            if CURRENT_MODE == "CUERPO":
                mask = mp_selfie.process(frame_rgb).segmentation_mask
            elif CURRENT_MODE == "CARAS":
                res = mp_face_mesh.process(frame_rgb)
                if res.multi_face_landmarks:
                    pts = np.array([(int(l.x*W_CAM), int(l.y*H_CAM)) for l in res.multi_face_landmarks[0].landmark])
                    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
            elif CURRENT_MODE == "PROFUNDIDAD":
                img_batch = midas_transforms(frame_rgb).to(device)
                with torch.no_grad():
                    pred = midas(img_batch)
                    pred = F.interpolate(pred.unsqueeze(1), size=(H_CAM, W_CAM), mode="bicubic").squeeze()
                d = pred.cpu().numpy()
                norm_d = (d - d.min()) / (d.max() - d.min() + 1e-6)
                mask = np.where(norm_d > Z_THRESHOLD, norm_d, 0.0) ** Z_EXP

            # 3. GPU Pipeline (PyTorch)
            frame_t = torch.from_numpy(frame).to(device).float().permute(2,0,1).unsqueeze(0) / 255.0
            flow_t = torch.from_numpy(flow).to(device)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(-1)
            
            flow_t[..., 0] /= (W_CAM / 2.0); flow_t[..., 1] /= (H_CAM / 2.0)
            flow_accumulator = (flow_accumulator + (flow_t * mask_t * 4.0).unsqueeze(0)) * 0.92
            
            # Difusión y Remapeo
            acc_b = TF.gaussian_blur(flow_accumulator.permute(0,3,1,2), [45, 45]).permute(0,2,3,1)
            distorted = F.grid_sample(frame_t, base_grid - acc_b, mode='bilinear', padding_mode='border', align_corners=True)

            # Salida
            out = (distorted.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            out = np.ascontiguousarray(out)
            
            fps = 1 / (time.time() - p_time)
            p_time = time.time()
            draw_hud(out, fps, f"CUDA ({device.type})")

            cv2.imshow("Kinematic Engine", out)
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()