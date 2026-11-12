# Pokemon Mirror（ポケモンミラー）

Webカメラで撮影した顔に似ているポケモンを、AIが探してくれるWebサービスです。

[日本語](#日本語) | [한국어](#한국어)

---

## 日本語

### 概要

ブラウザでWebカメラを起動して撮影すると、画像をAIサーバーに送り、CLIPモデルで顔の画像と各ポケモンの画像の類似度を計算して、最も似ているポケモンを表示します。
Web部分はSpring Boot、AI推論部分はPython（FastAPI）で分離した2サーバー構成です。

### 主な機能

- ブラウザ上でのWebカメラ撮影
- CLIPによる画像類似度の計算とマッチング
- マッチング結果の統計（どのポケモンが何回選ばれたか）
- レート制限による過剰リクエストの防止

### 技術スタック

| 区分 | 技術 |
|------|------|
| Web（mirror） | Java, Spring Boot, HTML, JavaScript |
| AI（ai） | Python, FastAPI, Uvicorn, Hugging Face Transformers（CLIP）, PyTorch（CPU）, OpenCV, NumPy, Pillow |
| Database | MySQL |
| Infra | Ubuntu Server, systemd, Cloudflare Tunnel, Tailscale |

### システム構成

~~
ブラウザ（Webカメラ）
   │ HTTPS
   ▼
Cloudflare Tunnel
   │
   ▼
mirror：Spring Boot（127.0.0.1 のみで待ち受け）
   ├─ AiClient ──▶ ai：FastAPI + CLIP（127.0.0.1:8091 のみで待ち受け）
   └─ StatsRepository ──▶ MySQL（統計データ）
~~

### ディレクトリ構成

~~
pokemon-mirror/
├── mirror/   Spring Boot（画面・API・統計・レート制限）
└── ai/       Python AIサーバー（CLIPによる類似度計算）
~~

### 工夫した点

- **WebとAIの分離**：重いAI推論をPythonサーバーに分け、Spring BootからHTTPで呼び出す構成にしました。それぞれ独立して再起動・リソース制限（systemdのMemoryMax、CPUQuota）ができます。
- **外部に公開するポートを最小限に**：Spring BootもAIサーバーも `127.0.0.1` でのみ待ち受け、外部からはCloudflare Tunnel経由でしかアクセスできません。
- **DB権限の最小化**：DBユーザーは接続元IPを限定し、必要な権限（SELECT, INSERT, UPDATE）のみを付与しています。
- **機密情報の管理**：DB接続情報はサーバー上の環境変数ファイルから読み込み、リポジトリには含めていません。
- **事前計算**：ポケモン画像の特徴量を `build_gallery.py` で事前に計算して保存し、リクエストごとの計算量を減らしています。

### ローカルでの実行方法

#### 1. AIサーバー

~~bash
cd ai
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m uvicorn pokemon_mirror_ai:app --host 127.0.0.1 --port 8091
~~

#### 2. データベース

`mirror/schema.sql` を参考にDBとユーザーを作成してください。

#### 3. Webサーバー

`mirror/src/main/resources/application.properties` の `${...}` に対応する環境変数（`DB_USERNAME`、`DB_PASSWORD` など）を設定してから実行します。

~~bash
cd mirror
mvn spring-boot:run
~~

### 注意事項

本プロジェクトは個人の学習・ポートフォリオ目的で制作した非公式のファン作品です。
ポケモンに関する名称・画像の権利は、株式会社ポケモン・任天堂・クリーチャーズ・ゲームフリークに帰属します。

---

## 한국어

### 개요

웹캠으로 찍은 얼굴과 닮은 포켓몬을 AI가 찾아주는 웹 서비스입니다. 웹은 Spring Boot, AI 추론은 Python(FastAPI) + CLIP으로 분리한 2서버 구성입니다.

### 주요 기능

- 브라우저 웹캠 촬영
- CLIP 기반 이미지 유사도 계산·매칭
- 매칭 결과 통계 (포켓몬별 선택 횟수)
- 레이트 리밋으로 과도한 요청 방지

### 기술 스택

Java / Spring Boot / Python / FastAPI / Transformers(CLIP) / PyTorch(CPU) / OpenCV / MySQL / Ubuntu Server / systemd / Cloudflare Tunnel / Tailscale

### 포인트

- 무거운 AI 추론을 Python 서버로 분리하고, 서비스별로 메모리·CPU 제한 설정
- Spring Boot와 AI 서버 모두 127.0.0.1에서만 대기, 외부 접근은 Cloudflare Tunnel로만 가능
- DB 사용자는 접속 IP를 제한하고 최소 권한(SELECT, INSERT, UPDATE)만 부여
- DB 접속 정보는 서버의 환경변수 파일에서 읽어 저장소에 포함하지 않음
- 포켓몬 이미지 특징량을 미리 계산해 두어 요청마다의 계산량 감소

### 주의

개인 학습·포트폴리오 목적의 비공식 팬 작품이며, 포켓몬 관련 명칭·이미지의 권리는 각 권리자에게 있습니다.
