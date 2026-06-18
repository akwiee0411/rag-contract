# model_server.py
"""
5090 模型伺服器
===============
本程式跑在 5090 機器上，以 FastAPI 對外暴露兩個模型的 HTTP API：

  POST /ocr
    輸入：multipart/form-data，欄位 file = JPEG/PNG 圖片
    輸出：JSON {"text": "<辨識結果>"}
    說明：使用本地 OCRFlux-3B 模型對單頁圖片執行 OCR

  POST /embed
    輸入：JSON {"texts": ["句子1", "句子2", ...]}
    輸出：JSON {"embeddings": [[float, ...], ...]}
    說明：使用本地 Yuan-embedding-2.0-zh 計算文字向量

  GET /health
    輸出：JSON {"ocr": "loaded"|"unloaded", "embed": "loaded"|"unloaded"}
    說明：查詢兩個模型的載入狀態

  POST /ocr/load   — 載入 OCR 模型到 GPU
  POST /ocr/unload — 卸載 OCR 模型，釋放 VRAM
  POST /embed/load   — 載入 Embedding 模型
  POST /embed/unload — 卸載 Embedding 模型

Ollama LLM 不需要包裝，直接由 Client 連線 Ollama 的原生 port 11434。

【啟動方式（5090 機器）】
  # 1. 先讓 Ollama 接受區網連線
  OLLAMA_HOST=0.0.0.0 ollama serve &

  # 2. 啟動本伺服器（預設 port 8000）
  python model_server.py

  # 或指定 port 和 OCR 模型路徑
  OCR_MODEL_PATH=/data/ocrmodel python model_server.py

【環境變數】
  OCR_MODEL_PATH   : OCRFlux-3B 模型目錄（預設 ./ocrmodel）
  MODEL_SERVER_PORT: 服務埠號（預設 8000）
  EMBED_DEVICE     : Embedding 計算裝置（預設 cpu，可設 cuda）
"""

import os
import gc
import io
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# ── 設定 ──────────────────────────────────────────────────────
OCR_MODEL_PATH    = os.environ.get("OCR_MODEL_PATH",
                                   r"C:\Users\user\Documents\rag_contract\ocrmodel")
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "8000"))
EMBED_DEVICE      = os.environ.get("EMBED_DEVICE", "cpu")

# ── ngrok 設定 ────────────────────────────────────────────────
# 設定 NGROK_AUTHTOKEN 環境變數即可啟用，不設定則跳過（純內網模式）
#
# 【啟動方式】
#   set NGROK_AUTHTOKEN=你的token    ← 只在當前視窗有效，不寫入磁碟
#   python model_server.py
#
# 【同時 tunnel Ollama（可選）】
#   set NGROK_OLLAMA_PORT=11434
#   python model_server.py
#
# 【切回內網模式】
#   不設定 NGROK_AUTHTOKEN，直接 python model_server.py 即可


# OCR prompt（與 doc_processor.py 保持一致）
EXTRACT_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "ALL tables should be presented in HTML format.\n"
    'If there are images or figures in the page, present them as "", '
    "(left,top,right,bottom) are the coordinates of the top-left and bottom-right corners.\n"
    "Present all titles and headings as H1 headings.\n"
    "Do not hallucinate.\n"
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("model_server")

# ── 全域模型變數 ───────────────────────────────────────────────
ocr_model      = None
ocr_tokenizer  = None
ocr_processor  = None
embed_model    = None   # sentence_transformers 或 HuggingFaceEmbedding 的底層模型


# ==========================================
# 模型載入 / 卸載
# ==========================================

def _load_ocr():
    global ocr_model, ocr_tokenizer, ocr_processor
    if ocr_model is not None:
        log.info("OCR 模型已載入，跳過")
        return
    log.info("載入 OCR 模型：%s", OCR_MODEL_PATH)
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        OCR_MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    ocr_model.eval()
    ocr_tokenizer = AutoTokenizer.from_pretrained(OCR_MODEL_PATH)
    ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_PATH)
    log.info("✅ OCR 模型載入完成")


def _unload_ocr():
    global ocr_model, ocr_tokenizer, ocr_processor
    if ocr_model is None:
        return
    try:
        import torch
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(ocr_model, recurse=True)
        except Exception:
            pass
        ocr_model.cpu()
        ocr_model = ocr_tokenizer = ocr_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("✅ OCR 模型已卸載")
    except Exception as e:
        log.error("OCR 卸載失敗：%s", e)


def _load_embed():
    global embed_model
    if embed_model is not None:
        log.info("Embedding 模型已載入，跳過")
        return
    log.info("載入 Embedding 模型（device=%s）", EMBED_DEVICE)
    # 使用 sentence_transformers 直接載入，不依賴 llama_index
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(
        "IEITYuan/Yuan-embedding-2.0-zh", device=EMBED_DEVICE
    )
    log.info("✅ Embedding 模型載入完成")


def _unload_embed():
    global embed_model
    if embed_model is None:
        return
    try:
        import torch
        embed_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("✅ Embedding 模型已卸載")
    except Exception as e:
        log.error("Embedding 卸載失敗：%s", e)


# ── 啟動時只載入 Embedding，OCR 等需要時再懶載入 ──────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Model Server 啟動中...")
    
    try:
        _load_embed()
    except Exception as e:
        log.warning("Embedding 模型啟動載入失敗（繼續啟動）：%s", e)
    yield
    log.info("🛑 Model Server 關閉，卸載模型...")
    _unload_ocr()
    _unload_embed()


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(title="5090 Model Server", lifespan=lifespan)


# ── OCR ───────────────────────────────────────────────────────

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    接收一張頁面圖片（JPEG/PNG），回傳 OCR 文字。

    Client 呼叫範例：
        import requests
        resp = requests.post(
            "http://5090_IP:8000/ocr",
            files={"file": open("page0.jpg", "rb")}
        )
        text = resp.json()["text"]
    """
    # OCR 懶載入
    if ocr_model is None:
        log.info("OCR 模型未載入，自動載入中...")
        try:
            _load_ocr()
        except Exception as e:
            raise HTTPException(status_code=503,
                                detail=f"OCR 模型載入失敗：{e}")

    try:
        img_bytes = await file.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"圖片讀取失敗：{e}")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  EXTRACT_PROMPT},
            ]},
        ]
        text = ocr_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = ocr_processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(ocr_model.device)
        output_ids = ocr_model.generate(
            **inputs, temperature=0.0, max_new_tokens=15000, do_sample=False
        )
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        result = ocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # 嘗試解析 JSON 格式的輸出（部分頁面）
        try:
            j = json.loads(result)
            result = j.get("natural_text", str(j))
        except Exception:
            pass

        return JSONResponse({"text": result})
    except Exception as e:
        log.error("OCR 推理失敗：%s", e)
        raise HTTPException(status_code=500, detail=f"OCR 推理失敗：{e}")


# ── Embedding ─────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
async def embed_endpoint(req: EmbedRequest):
    """
    接收一批文字，回傳對應的向量。

    Client 呼叫範例：
        import requests
        resp = requests.post(
            "http://5090_IP:8000/embed",
            json={"texts": ["合約第一條", "乙方應於期限內交付"]}
        )
        vectors = resp.json()["embeddings"]  # list[list[float]]
    """
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding 模型未載入")
    if not req.texts:
        return JSONResponse({"embeddings": []})
    try:
        vecs = embed_model.encode(req.texts, convert_to_numpy=True)
        return JSONResponse({"embeddings": vecs.tolist()})
    except Exception as e:
        log.error("Embedding 計算失敗：%s", e)
        raise HTTPException(status_code=500, detail=f"Embedding 計算失敗：{e}")


# ── 健康檢查 ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """回傳兩個模型的載入狀態，Client 啟動時可呼叫確認連線。"""
    return {
        "status": "ok",
        "ocr":    "loaded"   if ocr_model   is not None else "unloaded",
        "embed":  "loaded"   if embed_model is not None else "unloaded",
    }


# ── 記憶體管理端點（對應原本 Tab 5 的功能）────────────────────

@app.post("/ocr/load")
async def ocr_load():
    """載入 OCR 模型到 GPU。"""
    try:
        _load_ocr()
        return {"status": "ok", "ocr": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/unload")
async def ocr_unload():
    """卸載 OCR 模型，釋放 VRAM。"""
    _unload_ocr()
    return {"status": "ok", "ocr": "unloaded"}


@app.post("/embed/load")
async def embed_load():
    """載入 Embedding 模型。"""
    try:
        _load_embed()
        return {"status": "ok", "embed": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/unload")
async def embed_unload():
    """卸載 Embedding 模型，釋放記憶體。"""
    _unload_embed()
    return {"status": "ok", "embed": "unloaded"}







# ── 啟動 ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "model_server2:app",  # <-- 改成目前的檔名
        host="0.0.0.0",         
        port=MODEL_SERVER_PORT,
        reload=False,
        workers=1,              
    )