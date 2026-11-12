"""포켓몬 얼굴 도감(pokemon_face_gallery.pt)과 사람 얼굴 기준값(human_face_stats_v2.pt)을 만드는 스크립트.

pokemon-mirror-ai 폴더에서 실행하세요. 결과 파일도 같은 폴더에 저장됩니다.
  source venv/bin/activate
  pip install datasets
  nohup python build_gallery.py > build_gallery.log 2>&1 &
  tail -f build_gallery.log

GPU가 있으면 GPU로, 없으면 CPU로 돕니다. (CPU 기준 30분~1시간)
이미 만들어진 파일은 건너뜁니다. 다시 만들려면 해당 .pt 파일을 지우세요.
"""

import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, Owlv2ForObjectDetection, Owlv2Processor

BASE_DIR = Path(__file__).resolve().parent
GALLERY_PATH = BASE_DIR / "pokemon_face_gallery.pt"
HUMAN_STATS_PATH = BASE_DIR / "human_face_stats_v2.pt"

CLIP_NAME = "openai/clip-vit-base-patch32"
OWL_NAME = "google/owlv2-base-patch16-ensemble"
DATASET_NAME = "JJMack/pokemon-classification-gen1-9"
FAIRFACE_DATASET_NAME = "HuggingFaceM4/FairFace"
FAIRFACE_CONFIG = "0.25"   # 얼굴 주변 25% 여유로 잘린 버전 (웹캠 자르기 방식과 비슷함)
FAIRFACE_PER_GROUP = 150   # 인종 7 × 성별 2 × 150 = 2100장
MIN_AGE_LABEL = 2          # 0: 0-2세, 1: 3-9세는 빼고 10대 이상만 씁니다.
RACE_NAMES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

FACE_PROMPTS = ["the face of a cartoon creature", "the head of a cartoon creature"]
FACE_THRESHOLD = 0.15  # 이보다 자신 없으면 얼굴 대신 전체 그림을 씁니다.
FACE_MARGIN = 0.10     # 잘라낸 얼굴 주변에 남길 여유 비율
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def log(message):
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def to_rgb_on_white(image):
    """투명 배경 PNG를 흰 배경 위에 합성합니다."""
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).to(device).eval()
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)


@torch.inference_mode()
def embed_all(images, desc):
    """이미지 목록을 CLIP 벡터(길이 1로 정규화)로 바꿉니다. 결과는 CPU에 둡니다."""
    chunks = []
    for start in range(0, len(images), BATCH_SIZE):
        pixel_values = clip_processor(images=images[start:start + BATCH_SIZE], return_tensors="pt")["pixel_values"].to(device)
        features = clip_model(pixel_values=pixel_values).image_embeds
        chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        log(f"{desc}: {min(start + BATCH_SIZE, len(images))}/{len(images)}")
    return torch.cat(chunks)


def style_priority(file_name):
    """대표 이미지 우선순위. 숫자가 작을수록 우선이고, None이면 쓰지 않습니다."""
    stem = file_name.rsplit(".", 1)[0]
    lower = stem.lower()
    if "back" in lower:
        return None
    if stem.endswith("-"):
        return 0
    if lower.endswith("-home"):
        return 1
    return 2


def pick_representatives(dataset_all):
    """포켓몬마다 대표 이미지 1장의 번호를 고릅니다. 이로치는 뺍니다."""
    meta = dataset_all.select_columns(["file_name", "name", "shiny", "Type 1", "Type 2"])
    file_names = list(meta["file_name"])
    pokemon_names = list(meta["name"])
    shinies = list(meta["shiny"])
    type1s = list(meta["Type 1"])
    type2s = list(meta["Type 2"])

    best = {}
    for index, (file_name, name, shiny) in enumerate(zip(file_names, pokemon_names, shinies)):
        if shiny == "yes":
            continue
        priority = style_priority(file_name)
        if priority is None:
            continue
        if name not in best or priority < best[name][0]:
            best[name] = (priority, index)

    result = []
    for name, (priority, index) in best.items():
        types = [t for t in (type1s[index], type2s[index]) if t and t != "null"]
        result.append({
            "index": index,
            "name": name,
            "dex": int(file_names[index].split("-")[0]),
            "types": " / ".join(types),
        })
    result.sort(key=lambda entry: entry["dex"])
    return result


@torch.inference_mode()
def crop_face(image, owl_processor, owl_model):
    """OWLv2로 얼굴(머리)을 찾아 잘라냅니다. 못 찾으면 원본을 그대로 돌려줍니다."""
    inputs = owl_processor(text=[FACE_PROMPTS], images=image, return_tensors="pt").to(device)
    outputs = owl_model(**inputs)
    scores = torch.sigmoid(outputs.logits[0]).max(dim=1).values
    best = int(scores.argmax())
    if float(scores[best]) < FACE_THRESHOLD:
        return image, False

    cx, cy, w, h = outputs.pred_boxes[0, best].tolist()
    side = max(image.size)
    margin_w, margin_h = w * FACE_MARGIN, h * FACE_MARGIN
    x0 = max(0, int((cx - w / 2 - margin_w) * side))
    y0 = max(0, int((cy - h / 2 - margin_h) * side))
    x1 = min(image.width, int((cx + w / 2 + margin_w) * side))
    y1 = min(image.height, int((cy + h / 2 + margin_h) * side))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return image, False
    return image.crop((x0, y0, x1, y1)), True


def build_gallery():
    log("포켓몬 데이터셋 받는 중 (처음 한 번만 오래 걸립니다)")
    raw = load_dataset(DATASET_NAME)
    dataset_all = concatenate_datasets([raw["train"], raw["validation"], raw["test"]])
    entries = pick_representatives(dataset_all)
    log(f"대표 이미지 {len(entries)}종 선택 완료")

    owl_processor = Owlv2Processor.from_pretrained(OWL_NAME)
    owl_model = Owlv2ForObjectDetection.from_pretrained(OWL_NAME).to(device).eval()

    faces = []
    face_found = []
    for number, entry in enumerate(entries, start=1):
        image = to_rgb_on_white(dataset_all[entry["index"]]["image_data"])
        face, found = crop_face(image, owl_processor, owl_model)
        faces.append(face)
        face_found.append(found)
        if number % 25 == 0 or number == len(entries):
            log(f"얼굴 자르는 중: {number}/{len(entries)} (성공 {sum(face_found)})")
    del owl_model

    gallery = {
        "features": embed_all(faces, "도감 벡터 만드는 중"),
        "indices": [entry["index"] for entry in entries],
        "names": [entry["name"] for entry in entries],
        "dex": [entry["dex"] for entry in entries],
        "types": [entry["types"] for entry in entries],
        "face_found": face_found,
    }
    torch.save(gallery, GALLERY_PATH)
    log(f"도감 저장 완료: {GALLERY_PATH} (얼굴 찾기 성공 {sum(face_found)}/{len(face_found)}종)")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def find_face_like_site(image):
    """pokemon_mirror_ai.py의 find_face와 같은 규칙으로 얼굴 상자 (x0, y0, x1, y1)를 찾습니다. 없으면 None."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces_found = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces_found) == 0:
        return None
    x, y, w, h = (int(value) for value in max(faces_found, key=lambda face: face[2] * face[3]))
    margin = int(0.25 * w)
    return max(0, x - margin), max(0, y - margin), min(image.width, x + w + margin), min(image.height, y + h + margin)


def load_fairface():
    """FairFace 검증 세트(10,954장)를 받습니다. 기본 방식이 안 되면 허깅페이스 parquet 변환본으로 다시 시도합니다."""
    try:
        return load_dataset(FAIRFACE_DATASET_NAME, FAIRFACE_CONFIG, split="validation")
    except Exception as error:
        log(f"기본 방식으로 못 불러와서 parquet 변환본으로 다시 시도합니다: {error}")
        return load_dataset(FAIRFACE_DATASET_NAME, FAIRFACE_CONFIG, split="validation", revision="refs/convert/parquet")


def build_human_stats():
    # 인종 7그룹 × 성별 2 = 14그룹에서 똑같은 수만큼 뽑아 특정 인종·성별에 치우치지 않게 합니다.
    log("FairFace 얼굴 사진 받는 중 (인종·성별 균형 데이터셋, HuggingFaceM4/FairFace)")
    fairface = load_fairface()
    meta = fairface.select_columns(["race", "gender", "age"])
    races = list(meta["race"])
    genders = list(meta["gender"])
    ages = list(meta["age"])

    order = list(range(len(fairface)))
    random.Random(0).shuffle(order)
    target_total = len(RACE_NAMES) * 2 * FAIRFACE_PER_GROUP

    # 실제 사이트(pokemon_mirror_ai.py의 find_face)와 똑같이 Haar로 얼굴을 찾고 사방 25% 여유를 두고 자릅니다.
    # FairFace 사진은 이미 얼굴 위주로 잘려 있어서, Haar가 못 찾으면 사진 전체를 그대로 씁니다.
    human_images = []
    group_counts = {}
    haar_found = 0
    for index in order:
        if len(human_images) >= target_total:
            break
        if ages[index] < MIN_AGE_LABEL:
            continue
        group = (races[index], genders[index])
        if group_counts.get(group, 0) >= FAIRFACE_PER_GROUP:
            continue
        image = fairface[index]["image"].convert("RGB")
        box = find_face_like_site(image)
        if box is None:
            human_images.append(image)
        else:
            human_images.append(image.crop(box))
            haar_found += 1
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(human_images) % 500 == 0:
            log(f"사람 얼굴 자르는 중: {len(human_images)}/{target_total}")

    for race_id, race_name in enumerate(RACE_NAMES):
        male = group_counts.get((race_id, 0), 0)
        female = group_counts.get((race_id, 1), 0)
        log(f"    {race_name}: 남 {male}장 / 여 {female}장")
    log(f"사람 얼굴 {len(human_images)}장 준비 완료 (Haar로 다시 자른 사진 {haar_found}장)")

    human_features_all = embed_all(human_images, "사람 얼굴 벡터 만드는 중")
    order = torch.randperm(len(human_features_all), generator=torch.Generator().manual_seed(0))
    split = int(len(order) * 0.75)
    fit_features = human_features_all[order[:split]]
    test_features = human_features_all[order[split:]]

    gallery = torch.load(GALLERY_PATH, map_location="cpu")
    gallery_features = gallery["features"].float()
    human_center = fit_features.mean(dim=0)
    gallery_center = gallery_features.mean(dim=0)
    centered_gallery = torch.nn.functional.normalize(gallery_features - gallery_center, dim=-1)
    centered_fit = torch.nn.functional.normalize(fit_features - human_center, dim=-1)
    similarity = centered_fit @ centered_gallery.T
    std = similarity.std(dim=0)
    human_mean = similarity.mean(dim=0)
    human_std = std.clamp(min=float(std.median()))

    torch.save({
        "version": 2,
        "human_center": human_center,
        "gallery_center": gallery_center,
        "mean": human_mean,
        "std": human_std,
    }, HUMAN_STATS_PATH)
    log(f"사람 얼굴 기준값 저장 완료: {HUMAN_STATS_PATH}")

    # 검증: 따로 떼어 둔 사람 500명의 1등 포켓몬이 얼마나 골고루 나오는지
    centered_test = torch.nn.functional.normalize(test_features - human_center, dim=-1)
    for title, scores in (
        ("보정 없음", test_features @ gallery_features.T),
        ("센터링 보정", (centered_test @ centered_gallery.T - human_mean) / human_std),
    ):
        counts = torch.bincount(scores.argmax(dim=1), minlength=len(gallery["names"]))
        log(f"[{title}] 검증 {len(test_features)}명 → 서로 다른 1등 포켓몬 {int((counts > 0).sum())}종")
        for position in counts.topk(5).indices.tolist():
            share = 100 * int(counts[position]) / len(test_features)
            log(f"    No.{gallery['dex'][position]:04d} {gallery['names'][position]}: {int(counts[position])}명 ({share:.1f}%)")


if __name__ == "__main__":
    log(f"실행 장치: {device}")
    if GALLERY_PATH.exists():
        log("도감 파일이 이미 있어서 건너뜁니다.")
    else:
        build_gallery()
    if HUMAN_STATS_PATH.exists():
        log("사람 얼굴 기준값 파일이 이미 있어서 건너뜁니다.")
    else:
        build_human_stats()
    log("모두 완료")
