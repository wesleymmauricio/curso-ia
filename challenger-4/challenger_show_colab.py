import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
from collections import deque
import time
import random
import math

# --- Inicialização ---
model_yolo = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)  # Aumentado para 5
detector = dlib.cnn_face_detection_model_v1('/content/mmod_human_face_detector.dat')
# predictor = dlib.shape_predictor('/content/shape_predictor_68_face_landmarks.dat')

mp_drawing = mp.solutions.drawing_utils

# --- DICIONÁRIO DE MEMÓRIA POR ID ---
# Chave: ID da pessoa | Valor: deque com histórico
memorias_pessoas = {}

def detectar_careta(face_lms):
    # 1. Tensão nas Sobrancelhas (Distância entre sobrancelhas e olhos)
    sobrancelha_esq = face_lms[70].y 
    olho_esq = face_lms[159].y
    dist_sobrancelha = abs(sobrancelha_esq - olho_esq)

    # 2. Assimetria da Boca (Caretas laterais)
    canto_esq = face_lms[61]
    canto_dir = face_lms[291]
    nariz_centro = face_lms[1]
    
    dist_esq = abs(canto_esq.x - nariz_centro.x)
    dist_dir = abs(canto_dir.x - nariz_centro.x)
    assimetria_boca = abs(dist_esq - dist_dir)

    # 3. Critérios para Careta
    if assimetria_boca > 0.05: # Valor de exemplo, ajuste conforme necessário
        # return "Boca Torta / Careta"
        return True
    if dist_sobrancelha < 0.02:
        # return "Sobrancelha Franzida"
        return True
    
    return False


def analisar_dinamica_id(historico, frame, emocao_dominante, roi):
    # if len(historico) < 5: return None
    atividade = "Analisando"

    face_mesh_lms = None
    if mesh_res.multi_face_landmarks:
        face_mesh_lms = mesh_res.multi_face_landmarks[0].landmark

    # 1. YOLO
    # results_yolo = model_yolo(frame, conf=0.3, verbose=False)
    results_yolo = model_yolo(frame, conf=0.3, verbose=False)
    objetos_detectados = [model_yolo.names[int(c)] for r in results_yolo for c in r.boxes.cls]

    for r in results_yolo:
        for box in r.boxes:
            b = box.xyxy[0]
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)

    pulsos_esq_x = [f['pos'][0] for f in historico]
    pulsos_esq_y = [f['pos'][1] for f in historico]
    pulsos_dir_x = [f['pos'][2] for f in historico]
    pulsos_dir_y = [f['pos'][3] for f in historico]
    ombro_esq_y = [f['pos'][4] for f in historico]
    quadril_esq_y = [f['pos'][5] for f in historico]
    nariz_y = [f['pos'][6] for f in historico]
    ombro_dir_y = [f['pos'][7] for f in historico]
    ombro_esq_x = [f['pos'][8] for f in historico]
    quadril_dir_y = [f['pos'][9] for f in historico]
    quadril_dir_x = [f['pos'][10] for f in historico]
    nariz_x = [f['pos'][11] for f in historico]
    quadril_esq_x = [f['pos'][11] for f in historico]

    # Simples detecção de agitação rítmica (Dança/Aceno)
    variancia = np.std(pulsos_esq_x) + np.std(pulsos_esq_y)
    braco_elevado = np.mean(pulsos_esq_y) < np.mean(ombro_esq_y)
    quadril_std = np.std(quadril_esq_y)
    pulso_dir_y_std = np.std(pulsos_dir_y)
    pulso_esq_y_std = np.std(pulsos_esq_y)
    pulso_esq_x_std = np.std(pulsos_esq_x)
    pulso_dir_x_std = np.std(pulsos_dir_x)
    ombro_esq_y_std = np.std(ombro_esq_y)
    ombro_dir_y_std = np.std(ombro_dir_y)
    ombro_esq_x_std = np.std(ombro_esq_x)
    variancia_pulso = pulso_dir_y_std + pulso_esq_y_std

    pulso_dir_y_mean = np.mean(pulsos_dir_y)
    pulso_esq_y_mean = np.mean(pulsos_esq_y)
    pulso_esq_x_mean = np.mean(pulsos_esq_x)
    pulso_dir_x_mean = np.mean(pulsos_dir_x)
    nariz_mean = np.mean(nariz_y)
    ombro_dir_y_mean = np.mean(ombro_dir_y)

    # Centro do quadril
    quadril_x = (np.mean(quadril_esq_x) + np.mean(quadril_dir_x)) / 2
    quadril_y = (np.mean(quadril_esq_y) + np.mean(quadril_dir_y)) / 2

    # Calcula o ângulo em radianos e converte para graus
    delta_x = np.mean(nariz_x) - quadril_x
    delta_y = np.mean(nariz_y) - quadril_y
    angulo = math.degrees(math.atan2(delta_y, delta_x))

    ombros_y = ombro_dir_y_std + ombro_esq_y_std

    # --- 2. Dados da Face (Face Mesh) se disponível ---
    olhando_baixo = False
    if face_mesh_lms:
        testa = face_mesh_lms[10]
        nariz_f = face_mesh_lms[1]
        queixo = face_mesh_lms[152]
        p_sup = face_mesh_lms[159] # Pálpebra superior
        p_inf = face_mesh_lms[145] # Pálpebra inferior

        # Ratio de inclinação (Nariz descendo em direção ao queixo)
        ratio_head = abs(nariz_f.y - testa.y) / (abs(queixo.y - nariz_f.y) + 1e-6)
        # Abertura ocular (olhos "sumindo" ao olhar para baixo)
        abertura_olho = abs(p_sup.y - p_inf.y)
        
        if ratio_head > 1.25 or abertura_olho < 0.012:
            olhando_baixo = True
    
    # Prioridade A: Objetos Claros (YOLO)
    # if 'cell phone' in objetos: return "Uso de Celular"
    if 'laptop' in objetos_detectados: return "Trabalho (Laptop)"
    if 'cup' in objetos_detectados or 'bottle' in objetos_detectados:
        if pulso_dir_y_mean < nariz_mean or pulso_esq_y_mean < nariz_mean:
            return "Alimentacao / Bebendo"

    # Prioridade B: Leitura/Escrita (YOLO Book OU Geometria Facial)
    dancando_ = pulso_esq_x_std + pulso_dir_x_std + pulso_dir_y_std + pulso_esq_y_std + ombros_y + ombro_esq_x_std
    dist_entre_maos = abs(pulso_dir_x_mean - pulso_esq_x_mean)
    if 'book' in objetos_detectados or (olhando_baixo and pulso_dir_y_mean > ombro_dir_y_mean and dist_entre_maos < 0.2 and not dancando_):
        return "Lendo"
    
    if len(objetos_detectados) == 2 and all(item == "person" for item in objetos_detectados):
        return "Realizando Procedimento"
    # 2. Lógica para Dança (Movimento no corpo todo)
    # Variação no quadril (pulo/agacho) + variação nos pulsos

    dancando = dancando_ > 0.65
    # if pulso_esq_x_std > 0.06 and pulso_dir_x_std > 0.06:
    if abs(angulo) < 30 or abs(angulo) > 150:
        atividade = "Deitado"
    elif dancando:
        atividade = "Dancando"
    elif variancia > 0.06 and braco_elevado: 
        atividade = "Acenando"
    else:
        if face_mesh_lms:
            if esta_sorrindo_geometria(face_mesh_lms) and emocao_dominante in ['happy']:
                atividade = "Sorrindo"
            else:
                dist_mao_rosto = min(abs(pulso_dir_y_mean - nariz_mean), abs(pulso_esq_y_mean - nariz_mean))
                if dist_mao_rosto < 0.4:
                    atividade = "Maos no Rosto"
                elif emocao_dominante not in ['happy'] and detectar_careta(face_mesh_lms):
                    atividade = "Careta"
                elif emocao_dominante in ['neutral', 'happy']: return "Em repouso"
                else:
                    atividade = "Anomalia"
    # elif variancia > 0.06: atividade = "Atividade Dinamica"
        
    mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return atividade


def categorizar_multpessoa(pose_lm, mesh_lm, emocao, objetos, dinamica):
    if not pose_lm: return "Indeterminado"
    if dinamica: return dinamica

    # Reutiliza sua lógica de inclinação de cabeça e objetos...
    # (Omitido aqui por brevidade, mas permanece igual à V11)
    return "Ativo" if 'cell phone' in objetos else "Em repouso"


def esta_sorrindo_geometria(face_lms):
    # Pontos dos cantos da boca
    canto_esq = face_lms[61]
    canto_dir = face_lms[291]
    labio_sup = face_lms[0]
    labio_inf = face_lms[17]

    # 1. Distância horizontal (largura da boca)
    largura_boca = abs(canto_dir.x - canto_esq.x)

    # 2. Distância vertical (abertura da boca)
    abertura_boca = abs(labio_inf.y - labio_sup.y)

    # 3. Ratio de sorriso: se a boca está larga mas não muito aberta
    # O valor 0.25 é um limiar (threshold) que você pode ajustar
    if largura_boca > 0.08: 
        return True
    return False

def detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi):
    atividade = "Analisando"
    if pose_res.pose_landmarks:
        p_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        p_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        ombro_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        ombro_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        quadril_esq = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        quadril_dir = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        nariz = pose_res.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        memorias_pessoas[track_id].append({'pos': (p_esq.x, p_esq.y, p_dir.x, p_dir.y, ombro_esq.y, quadril_esq.y, nariz.y, ombro_dir.y, 
                                                   ombro_esq.x, quadril_dir.y, quadril_dir.x, nariz.x, quadril_esq.x)})
        atividade = analisar_dinamica_id(memorias_pessoas[track_id], frame, emocao_dominante, roi)

    else:
        print("Nao ha pose")
    
    return atividade

cap = cv2.VideoCapture("/content/unlocking_facial_recognition_diverse_activities_analysis.mp4")
frame_count = 0

colors = []
# color = tuple(random.randint(0, 255) for _ in range(3))
colors.append((255, 0, 0))
colors.append((255, 225, 0))
colors.append((60, 222, 249))
colors.append((60, 213, 11))
colors.append((255, 253, 252))
colors.append((236, 106, 102))
colors.append((236, 106, 0))
colors.append((236, 0, 232))
colors.append((0, 122, 232))
colors.append((100, 100, 100))
colors.append((0, 0, 0))
colors.append((10, 200, 200))

relatorio_dados = {
    "ids_detectados": set(),
    "atividades_vistas": set(),
    "emocoes_vistas": set(),
    "total_anomalias": 0 
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    atividade = None
    emocao_dominante = "neutral"
    
    # if frame_count % 3 != 0: continue

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. YOLO TRACKING (Essencial para múltiplas pessoas)
    results_yolo = model_yolo.track(frame_rgb, persist=True, conf=0.5, verbose=False, iou=0.4)
    # results_yolo = model_yolo.track(frame, persist=True, conf=0.5, verbose=False, iou=0.4)

    # 2. MediaPipe Pose e Mesh (Processam o frame todo)
    pose_res = pose.process(frame_rgb)
    mesh_res = face_mesh.process(frame_rgb)

    if results_yolo[0].boxes.id is not None:
        ids = results_yolo[0].boxes.id.int().cpu().tolist()
        boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
        classes = results_yolo[0].boxes.cls.int().cpu().tolist()

        for i, track_id in enumerate(ids):
            # print(f"classe: {classes[i]}")
            if classes[i] != 0: continue  # Focar apenas em 'person' (classe 0)

            # if track_id > 4: continue

            # Criar memória para novo ID se não existir
            if track_id not in memorias_pessoas:
                memorias_pessoas[track_id] = deque(maxlen=10)

            # Coordenadas da pessoa atual
            x1, y1, x2, y2 = boxes[i]

            # 4. DeepFace (Apenas na região da box desta pessoa)
            emocao = "neutral"
            try:
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                if roi.size > 0:
                    face = detector(roi, 0)[0]
                    # for face in faces:
                    xf1, yf1, xf2, yf2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()

                    # 4. Emoção (DeepFace)
                    emocao_dominante = "neutral"
                    try:
                        roi_face = roi[max(0, yf1):yf2, max(0, xf1):xf2]
                        if roi_face.size > 0:
                            objs_df = DeepFace.analyze(roi_face, actions=['emotion'], enforce_detection=False)
                            emocao_dominante = objs_df[0]['dominant_emotion']
                    except:
                        print("Falha")
                        pass
                    
                    # # 3. Filtrar Pose para esta pessoa específica
                    # (Aqui simplificamos: se o pulso está dentro da box da pessoa, pertence a ela)
                    atividade = detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi)
                    
                    # --- COLETA DE DADOS PARA O RESUMO ---
                    if atividade == "Anomalia":
                        relatorio_dados["total_anomalias"] += 1

                    # --- COLETA DE DADOS PARA O RESUMO ---
                    relatorio_dados["ids_detectados"].add(track_id)
                    if atividade: relatorio_dados["atividades_vistas"].add(atividade)
                    if emocao_dominante: relatorio_dados["emocoes_vistas"].add(emocao_dominante)
                    # -------------------------------------

                    cv2.rectangle(roi, (xf1, yf1), (xf2, yf2), colors[i], 2)
                    cv2.putText(roi, f"Id: {track_id}", (xf1, yf2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                colors[i], 2)
                    cv2.putText(roi, f"Emot: {emocao_dominante}", (xf1, yf2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                colors[i], 2)

            except:
                atividade = detecta_atividade(pose_res, mp_pose, memorias_pessoas, frame, mesh_res, emocao_dominante, track_id, roi)

            # UI por pessoa
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)
            cv2.putText(frame, f"ID:{track_id}", (int(x1) + 10, int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        colors[i], 2)
            cv2.putText(frame, f"Act: {atividade}", (int(x1) + 10, int(y1) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                colors[i], 2)
            
    clear_output(wait=True)
    cv2_imshow(frame)

cap.release()

with open("resumo_analise.txt", "w", encoding="utf-8") as f:
    f.write("=== RESUMO AUTOMÁTICO DE ANÁLISE DE VÍDEO ===\n")
    f.write(f"Total de frames analisados: {frame_count}\n") # <--- Adicionado
    f.write(f"Total de pessoas únicas identificadas: {len(relatorio_dados['ids_detectados'])}\n")
    f.write(f"IDs detectados: {sorted(list(relatorio_dados['ids_detectados']))}\n")
    f.write(f"Atividades principais observadas: {', '.join(relatorio_dados['atividades_vistas'])}\n")
    f.write(f"Emoções detectadas: {', '.join(relatorio_dados['emocoes_vistas'])}\n")
    f.write(f"Número de anomalias detectadas: {relatorio_dados['total_anomalias']}\n") # <--- Adicionado
    f.write("============================================\n")

print("Relatório 'resumo_analise.txt' gerado com sucesso!")