import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace
# from google.colab.patches import cv2_imshow # Removido para otimização
from IPython.display import clear_output
from collections import deque
import time
import random
import math
from tqdm import tqdm 

# --- Inicialização ---
model_yolo = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)
detector = dlib.cnn_face_detection_model_v1('/content/mmod_human_face_detector.dat')

mp_drawing = mp.solutions.drawing_utils
memorias_pessoas = {}

# (As funções detectar_careta, analisar_dinamica_id, categorizar_multpessoa, 
# esta_sorrindo_geometria e detecta_atividade permanecem EXATAMENTE iguais)

def detectar_careta(face_lms):
    sobrancelha_esq = face_lms[70].y 
    olho_esq = face_lms[159].y
    dist_sobrancelha = abs(sobrancelha_esq - olho_esq)
    canto_esq = face_lms[61]
    canto_dir = face_lms[291]
    nariz_centro = face_lms[1]
    dist_esq = abs(canto_esq.x - nariz_centro.x)
    dist_dir = abs(canto_dir.x - nariz_centro.x)
    assimetria_boca = abs(dist_esq - dist_dir)
    if assimetria_boca > 0.05: return True
    if dist_sobrancelha < 0.02: return True
    return False

def analisar_dinamica_id(historico, frame, emocao_dominante, roi):
    atividade = "Analisando"
    face_mesh_lms = None
    if mesh_res.multi_face_landmarks:
        face_mesh_lms = mesh_res.multi_face_landmarks[0].landmark
    results_yolo = model_yolo(frame, conf=0.3, verbose=False)
    objetos_detectados = [model_yolo.names[int(c)] for r in results_yolo for c in r.boxes.cls]
    for r in results_yolo:
        for box in r.boxes:
            b = box.xyxy[0]
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
    pulsos_esq_x = [f['pos'][0] for f in historico]; pulsos_esq_y = [f['pos'][1] for f in historico]
    pulsos_dir_x = [f['pos'][2] for f in historico]; pulsos_dir_y = [f['pos'][3] for f in historico]
    ombro_esq_y = [f['pos'][4] for f in historico]; quadril_esq_y = [f['pos'][5] for f in historico]
    nariz_y = [f['pos'][6] for f in historico]; ombro_dir_y = [f['pos'][7] for f in historico]
    ombro_esq_x = [f['pos'][8] for f in historico]; quadril_dir_y = [f['pos'][9] for f in historico]
    quadril_dir_x = [f['pos'][10] for f in historico]; nariz_x = [f['pos'][11] for f in historico]
    quadril_esq_x = [f['pos'][11] for f in historico]
    variancia = np.std(pulsos_esq_x) + np.std(pulsos_esq_y)
    braco_elevado = np.mean(pulsos_esq_y) < np.mean(ombro_esq_y)
    pulso_dir_y_std = np.std(pulsos_dir_y)
    pulso_esq_y_std = np.std(pulsos_esq_y)
    pulso_esq_x_std = np.std(pulsos_esq_x)
    pulso_dir_x_std = np.std(pulsos_dir_x)
    ombro_esq_y_std = np.std(ombro_esq_y)
    ombro_dir_y_std = np.std(ombro_dir_y)
    ombro_esq_x_std = np.std(ombro_esq_x)
    ombros_y = ombro_dir_y_std + ombro_esq_y_std

    pulso_dir_y_mean = np.mean(pulsos_dir_y)
    pulso_esq_y_mean = np.mean(pulsos_esq_y)
    pulso_esq_x_mean = np.mean(pulsos_esq_x)
    pulso_dir_x_mean = np.mean(pulsos_dir_x)
    nariz_mean = np.mean(nariz_y)
    ombro_dir_y_mean = np.mean(ombro_dir_y)

    quadril_x = (np.mean(quadril_esq_x) + np.mean(quadril_dir_x)) / 2
    quadril_y = (np.mean(quadril_esq_y) + np.mean(quadril_dir_y)) / 2
    delta_x = np.mean(nariz_x) - quadril_x; delta_y = np.mean(nariz_y) - quadril_y

    angulo = math.degrees(math.atan2(delta_y, delta_x))
    olhando_baixo = False
    if face_mesh_lms:
        testa = face_mesh_lms[10]; nariz_f = face_mesh_lms[1]; queixo = face_mesh_lms[152]
        p_sup = face_mesh_lms[159]; p_inf = face_mesh_lms[145]
        ratio_head = abs(nariz_f.y - testa.y) / (abs(queixo.y - nariz_f.y) + 1e-6)
        abertura_olho = abs(p_sup.y - p_inf.y)
        if ratio_head > 1.25 or abertura_olho < 0.012: olhando_baixo = True

    if 'laptop' in objetos_detectados: return "Trabalho (Laptop)"
    if 'cup' in objetos_detectados or 'bottle' in objetos_detectados:
        if pulso_dir_y_mean < nariz_mean or pulso_esq_y_mean < nariz_mean: return "Alimentacao / Bebendo"
    
    dancando_ = pulso_esq_x_std + pulso_dir_x_std + pulso_dir_y_std + pulso_esq_y_std + ombros_y + ombro_esq_x_std
    dist_entre_maos = abs(pulso_dir_x_mean - pulso_esq_x_mean)
    
    if 'book' in objetos_detectados or (olhando_baixo and pulso_dir_y_mean > ombro_dir_y_mean and dist_entre_maos < 0.2 and not dancando_): return "Lendo"
    if len(objetos_detectados) == 2 and all(item == "person" for item in objetos_detectados): return "Realizando Procedimento"
    dancando = dancando_ > 0.65
    if abs(angulo) < 30 or abs(angulo) > 150: atividade = "Deitado"
    elif dancando: atividade = "Dancando"
    elif variancia > 0.06 and braco_elevado: atividade = "Acenando"
    else:
        if face_mesh_lms:
            if esta_sorrindo_geometria(face_mesh_lms) and emocao_dominante in ['happy']: atividade = "Sorrindo"
            else:
                dist_mao_rosto = min(abs(pulso_dir_y_mean - nariz_mean), abs(pulso_esq_y_mean - nariz_mean))
                if dist_mao_rosto < 0.4: atividade = "Maos no Rosto"
                elif emocao_dominante not in ['happy'] and detectar_careta(face_mesh_lms): atividade = "Careta"
                elif emocao_dominante in ['neutral', 'happy']: return "Em repouso"
                else: atividade = "Anomalia"
    mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return atividade

def esta_sorrindo_geometria(face_lms):
    largura_boca = abs(face_lms[291].x - face_lms[61].x)
    return largura_boca > 0.08

def detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi):
    atividade = "Analisando"
    if pose_res.pose_landmarks:
        p_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        p_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        o_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        o_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        q_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        q_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        nariz = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        memorias_pessoas[track_id].append({'pos': (p_esq.x, p_esq.y, p_dir.x, p_dir.y, o_esq.y, q_esq.y, nariz.y, o_dir.y, o_esq.x, q_dir.y, q_dir.x, nariz.x, q_esq.x)})
        atividade = analisar_dinamica_id(memorias_pessoas[track_id], frame, emocao_dominante, roi)
    return atividade

# --- CONFIGURAÇÃO DE VÍDEO E PROGRESSO ---
input_path = "/content/unlocking_facial_recognition_diverse_activities_analysis.mp4"
output_path = "/content/resultado_analise.mp4"

cap = cv2.VideoCapture(input_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Definir o codec e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

colors = [(255, 0, 0), (255, 225, 0), (60, 222, 249), (60, 213, 11), (255, 253, 252), (236, 106, 102), (236, 106, 0), (236, 0, 232), (0, 122, 232), (100, 100, 100), (0, 0, 0), (10, 200, 200)]

relatorio_dados = {
    "ids_detectados": set(),
    "atividades_vistas": set(),
    "emocoes_vistas": set()
}

# Barra de progresso tqdm
pbar = tqdm(total=total_frames, desc="Processando Vídeo")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_yolo = model_yolo.track(frame_rgb, persist=True, conf=0.5, verbose=False, iou=0.4)
    pose_res = pose.process(frame_rgb)
    mesh_res = face_mesh.process(frame_rgb)

    if results_yolo[0].boxes.id is not None:
        ids = results_yolo[0].boxes.id.int().cpu().tolist()
        boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
        classes = results_yolo[0].boxes.cls.int().cpu().tolist()

        for i, track_id in enumerate(ids):
            if classes[i] != 0: continue
            if track_id not in memorias_pessoas:
                memorias_pessoas[track_id] = deque(maxlen=10)

            x1, y1, x2, y2 = boxes[i]
            emocao_dominante = "neutral"
            atividade = "Analisando"
            
            try:
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                if roi.size > 0:
                    face_detects = detector(roi, 0)
                    if len(face_detects) > 0:
                        face = face_detects[0]
                        xf1, yf1, xf2, yf2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
                        roi_face = roi[max(0, yf1):yf2, max(0, xf1):xf2]
                        if roi_face.size > 0:
                            objs_df = DeepFace.analyze(roi_face, actions=['emotion'], enforce_detection=False, silent=True)
                            emocao_dominante = objs_df[0]['dominant_emotion']
                        
                        cv2.rectangle(roi, (xf1, yf1), (xf2, yf2), colors[i % len(colors)], 2)
                        cv2.putText(roi, f"Id: {track_id}", (xf1, yf2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % len(colors)], 2)
                        cv2.putText(roi, f"Emot: {emocao_dominante}", (xf1, yf2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % len(colors)], 2)
                
                atividade = detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi)
                
            except Exception as e:
                atividade = detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi)

            # Resumo e UI
            relatorio_dados["ids_detectados"].add(track_id)
            if atividade: relatorio_dados["atividades_vistas"].add(atividade)
            if emocao_dominante: relatorio_dados["emocoes_vistas"].add(emocao_dominante)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[i % len(colors)], 2)
            cv2.putText(frame, f"ID:{track_id} Act:{atividade}", (int(x1) + 10, int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % len(colors)], 2)

    out.write(frame) # Salva o frame no novo arquivo
    pbar.update(1)   # Atualiza barra de progresso

pbar.close()
cap.release()
out.release()

# --- GERAÇÃO DO TXT ---
with open("resumo_analise.txt", "w", encoding="utf-8") as f:
    f.write("=== RESUMO AUTOMÁTICO DE ANÁLISE DE VÍDEO ===\n")
    f.write(f"Total de pessoas únicas identificadas: {len(relatorio_dados['ids_detectados'])}\n")
    f.write(f"Atividades principais observadas: {', '.join(relatorio_dados['atividades_vistas'])}\n")
    f.write(f"Emoções detectadas: {', '.join(relatorio_dados['emocoes_vistas'])}\n")
    f.write("============================================\n")
