"""포켓몬 닮은꼴 AI 서버 (CPU 전용, GPU 불필요) - Spring Boot(pokemon-mirror)가 내부에서만 호출합니다.

같은 폴더에 아래 파일이 있어야 합니다.
  - pokemon_face_gallery.pt   : 포켓몬 얼굴 도감
  - human_face_stats_v2.pt    : 사람 얼굴 기준값, 센터링 보정 버전 (없으면 보정 없이 동작)

실행:
  uvicorn pokemon_mirror_ai:app --host 127.0.0.1 --port 8091
  (외부에 열지 마세요. 웹페이지와 요청 제한은 Spring Boot가 담당합니다.)

웹캠 프레임은 메모리에서만 처리하고 어디에도 저장하지 않습니다.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from starlette.concurrency import run_in_threadpool
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

BASE_DIR = Path(__file__).resolve().parent
GALLERY_PATH = BASE_DIR / "pokemon_face_gallery.pt"
HUMAN_STATS_PATH = BASE_DIR / "human_face_stats_v2.pt"

CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_UPLOAD_BYTES = 300_000  # 320px JPEG 한 장은 보통 10~30KB입니다.
TOP_K = 10                  # 브라우저가 최근 몇 프레임 결과를 평균 낼 수 있게 넉넉히 돌려줍니다.
CPU_THREADS = 2             # 홈서버 다른 서비스가 느려지면 2로 줄이세요.

torch.set_num_threads(CPU_THREADS)

clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).eval()
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)

gallery = torch.load(GALLERY_PATH, map_location="cpu")
gallery_features = gallery["features"].float()

if HUMAN_STATS_PATH.exists():
    human_stats = torch.load(HUMAN_STATS_PATH, map_location="cpu")
    # 사람 사진 공통 성분과 포켓몬 그림 공통 성분을 각각 빼서 생김새 차이만 비교합니다.
    human_center = human_stats["human_center"].float()
    gallery_center = human_stats["gallery_center"].float()
    search_features = torch.nn.functional.normalize(gallery_features - gallery_center, dim=-1)
    human_mean = human_stats["mean"].float()
    human_std = human_stats["std"].float()  # 파일을 만들 때 이미 중간값으로 바닥을 깔아 두었습니다.
else:
    human_center = None
    search_features = gallery_features
    human_mean = None
    human_std = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model_lock = threading.Lock()

app = FastAPI(title="Pokemon Mirror AI", docs_url=None, redoc_url=None, openapi_url=None)


@torch.inference_mode()
def embed(image):
    """얼굴 이미지 1장을 CLIP 벡터(길이 1로 정규화)로 바꿉니다."""
    pixel_values = clip_processor(images=[image], return_tensors="pt")["pixel_values"]
    features = clip_model(pixel_values=pixel_values).image_embeds[0]
    return torch.nn.functional.normalize(features, dim=-1)


def find_face(frame):
    """가장 큰 얼굴을 찾아 여유를 조금 둔 상자 (x0, y0, x1, y1)를 돌려줍니다. 없으면 None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_found = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces_found) == 0:
        return None
    x, y, w, h = (int(value) for value in max(faces_found, key=lambda face: face[2] * face[3]))
    margin = int(0.25 * w)
    height, width = frame.shape[:2]
    return max(0, x - margin), max(0, y - margin), min(width, x + w + margin), min(height, y + h + margin)


def analyze(body):
    """JPEG 바이트를 받아 얼굴을 찾고 닮은 포켓몬 목록을 돌려줍니다."""
    frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

    box = find_face(frame)
    if box is None:
        return {"found": False}

    x0, y0, x1, y1 = box
    face = Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    with model_lock:
        query = embed(face)

    if human_center is not None:
        query = torch.nn.functional.normalize(query - human_center, dim=-1)
    scores = search_features @ query
    if human_mean is not None:
        scores = (scores - human_mean) / human_std
    values, positions = scores.topk(TOP_K)

    height, width = frame.shape[:2]
    return {
        "found": True,
        "box": [x0 / width, y0 / height, x1 / width, y1 / height],
        "matches": [
            {
                "dex": gallery["dex"][position],
                "name": gallery["names"][position],
                "types": gallery["types"][position],
                "score": round(value, 4),
            }
            for value, position in zip(values.tolist(), positions.tolist())
        ],
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "pokemon": len(gallery["names"]), "human_stats": human_mean is not None}


@app.post("/match")
async def match(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="이미지가 비어 있습니다.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지가 너무 큽니다.")
    return await run_in_threadpool(analyze, body)
