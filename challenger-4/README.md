# 🎥 Sistema de Análise Comportamental e Reconhecimento Facial Multi-Modal

Este projeto implementa um sistema avançado de **análise comportamental e reconhecimento facial**, utilizando múltiplos modelos de **Inteligência Artificial** para identificar pessoas, rastrear indivíduos, detectar emoções e inferir atividades humanas a partir de vídeos.

A execução foi projetada para o ambiente **Google Colab**, garantindo facilidade de uso e reprodutibilidade.

---

## 📌 Visão Geral (Overview)

O sistema realiza, de forma integrada:

- Detecção e rastreamento de pessoas
- Reconhecimento facial
- Análise de emoções
- Extração de biometria corporal
- Inferência de atividades humanas

O pipeline combina modelos de visão computacional e deep learning para produzir um **vídeo final anotado** e um **arquivo de resumo da análise**.

---

## 🧰 Tecnologias e Bibliotecas Utilizadas

### 🔹 Ultralytics (YOLOv8)
- Detecção e rastreamento (tracking) de pessoas e objetos
- Atribuição de **ID único** por indivíduo para consistência entre frames

### 🔹 MediaPipe (Google)
- **Pose:** extração de pontos do esqueleto corporal (braços, ombros, quadris, etc.)
- **FaceMesh:** mapeamento de **468 pontos faciais** para análise de micro-expressões

### 🔹 DeepFace
- Análise facial profunda
- Classificação da **emoção dominante** (feliz, neutro, triste, etc.)

### 🔹 Dlib
- Detector facial secundário de alta precisão
- Uso do modelo **MMOD** para garantir recortes faciais (ROI) mais confiáveis

### 🔹 OpenCV (cv2)
- Manipulação de vídeo
- Desenho de bounding boxes e textos
- Conversão de cores (BGR ↔ RGB)
- Geração do vídeo final

---

## 🚀 Como Executar o Projeto

### 1️⃣ Abrir o Google Colab

Configure o ambiente conforme abaixo:

- **Linguagem:** Python 3  
- **Hardware Accelerator:** CPU  
- **Versão:** Latest  

---

### 2️⃣ Upload dos Arquivos Necessários

Faça o upload manual dos seguintes arquivos no Google Colab:

#### 🎬 Vídeo de entrada
- `unlocking_facial_recognition_diverse_activities_analysis.mp4`

> ⚠️ **Observação:**  
> O vídeo não está disponível no repositório Git devido ao seu tamanho.  
> Realize o download a partir da fonte disponibilizada pela instituição e **renomeie exatamente** conforme definido no código.

#### 📄 Modelo de detecção facial
- `mmod_human_face_detector.dat`

---

### 3️⃣ Instalação das Dependências

Execute os comandos abaixo no Colab:

```bash
!pip uninstall -y mediapipe
!pip install --no-cache-dir mediapipe==0.10.14 deepface ultralytics
```

### 4️⃣ Execução do Código

O projeto possui **duas versões de execução**, cada uma com um objetivo específico:

#### ▶️ challenger_show_colab.py
- Exibe os frames do vídeo em tempo real no ambiente do **Google Colab**
- Apresenta as detecções, identificações e emoções diretamente na tela

#### 🎞️ challenger_gera_video.py
- Gera um vídeo final chamado:
  - `resultado_analise.mp4`
- O vídeo contém toda a análise de reconhecimento facial e comportamental das pessoas presentes no vídeo de entrada

---

## 📊 Resultados Gerados

Ao final da execução, são produzidos os seguintes artefatos:

- 🎥 **Vídeo final com análise e reconhecimento facial**
  - Arquivo: `resultado_analise.mp4`

- 📄 **Arquivo de resumo da análise**
  - Arquivo: `resumo_analise.txt`

---

## 🔗 Vídeo de Demonstração

O vídeo de domonstração também está disponível no OneDrive:

https://1drv.ms/v/c/ff7c96d3b1848b0a/IQA7_ZoLWj3zSYob-sYmMDcDAcqYSThaTl2dBwZspPyPR_M?e=AqV2sX

Além do video, também disponibilizamos o resumo gerado durante os testes resumo_analise.txt

---


