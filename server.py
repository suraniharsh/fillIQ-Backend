import io
import sys
import os
import re
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    TextStreamer,
    pipeline as hf_pipeline,
)
from tqdm import tqdm

# Directory to save OCR results
OUTPUT_DIR = Path("ocr_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "lightonai/LightOnOCR-2-1B"
JSON_MODEL_ID = os.getenv("JSON_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = 512

torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))

app = FastAPI(title="LightOnOCR CPU Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.1.33:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- global singletons ----
processor = None
model = None
json_tokenizer = None
json_model = None
json_llm_chain = None

# The exact flat schema we want from the helper LLM
EMPTY_CONTACT = {
    "salutation": "",
    "first_name": "",
    "last_name": "",
    "gender": "",
    "email_id": "",
    "mobile_no": "",
    "no_of_employees": "",
    "country": "",
    "company_name": "",
    "title": "",
    "industry": "",
    "city": "",
    "state": "",
    "status": "",
}

# simple in-process queue (prevents CPU dogpile)
request_lock = asyncio.Lock()


class ProgressStreamer(TextStreamer):
    """Streamer that shows a progress bar during token generation."""

    def __init__(self, tokenizer, max_new_tokens, show_progress=True, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.show_progress = show_progress and sys.stdout.isatty()
        self.pbar = None
        if self.show_progress:
            self.pbar = tqdm(total=max_new_tokens, desc="🧠 Generating", unit="tok", ncols=80)
        self.token_count = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.token_count += 1
        if self.pbar:
            self.pbar.update(1)
            if stream_end:
                self.pbar.close()

    def end(self):
        super().end()
        if self.pbar:
            self.pbar.close()


def _strip_fences(text: str) -> str:
    """Remove markdown code fences and leading 'Assistant:' tags."""
    text = re.sub(r"^\s*Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def extract_json_block(text: str) -> dict | None:
    """Grab the first JSON object embedded in a string."""
    text = _strip_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_contact(raw_dict: dict | None) -> dict:
    """Force any dict into the flat contact schema with empty-string defaults."""
    result = {k: "" for k in EMPTY_CONTACT}
    if not raw_dict or not isinstance(raw_dict, dict):
        return result
    for key in EMPTY_CONTACT:
        val = raw_dict.get(key)
        if val is not None and val != "" and not isinstance(val, (dict, list)):
            result[key] = str(val).strip()
    return result


def _coerce_helper_output(helper_result) -> str:
    """Normalize LangChain/HF outputs into plain text."""
    if helper_result is None:
        return ""
    if isinstance(helper_result, str):
        return helper_result
    if isinstance(helper_result, dict):
        if "generated_text" in helper_result:
            return helper_result["generated_text"]
        if "text" in helper_result:
            return helper_result["text"]
    content = getattr(helper_result, "content", None)
    if isinstance(content, str):
        return content
    return str(helper_result)


def generate_contact_json(raw_text: str) -> tuple[dict, str | None]:
    """Use Qwen helper to convert OCR text → flat contact JSON."""
    if json_llm_chain is None:
        return normalize_contact(None), None

    helper_raw_text = None
    try:
        helper_result = json_llm_chain.invoke({"raw_text": raw_text})
        helper_raw_text = _coerce_helper_output(helper_result).strip()
        if not helper_raw_text:
            return normalize_contact(None), None
        parsed = extract_json_block(helper_raw_text)
        return normalize_contact(parsed), helper_raw_text
    except Exception as exc:
        print(f"⚠️  Helper LLM error: {exc}")
        parsed = extract_json_block(helper_raw_text or "")
        return normalize_contact(parsed), helper_raw_text




# Concrete example the 0.5B model can copy
_EXAMPLE_JSON = json.dumps(EMPTY_CONTACT, indent=2)


@app.on_event("startup")
async def load_model():
    global processor, model, json_tokenizer, json_model, json_llm_chain

    print("🔹 Loading processor...")
    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)

    print("🔹 Loading model (CPU)... this will take time")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✅ Model ready")

    print(f"🔹 Loading JSON helper model ({JSON_MODEL_ID})...")
    helper_on_cuda = torch.cuda.is_available()
    helper_dtype = torch.float16 if helper_on_cuda else torch.float32

    json_tokenizer = AutoTokenizer.from_pretrained(
        JSON_MODEL_ID,
        trust_remote_code=True,
    )
    if json_tokenizer.pad_token is None:
        json_tokenizer.pad_token = json_tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": helper_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if helper_on_cuda:
        model_kwargs["device_map"] = "auto"

    json_model = AutoModelForCausalLM.from_pretrained(
        JSON_MODEL_ID,
        **model_kwargs,
    )
    json_model.eval()

    # Simple, concrete prompt the 0.5B model can actually follow
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You extract contact info from business card text into JSON. "
                    "Reply with ONLY a JSON object, no extra text.\n"
                    "Use EXACTLY these keys (use empty string \"\" when not found):\n"
                    f"{_EXAMPLE_JSON}"
                ),
            ),
            (
                "user",
                "Business card text:\n{raw_text}\n\nJSON:",
            ),
        ]
    )

    pipeline_device = 0 if helper_on_cuda else -1

    generation_pipeline = hf_pipeline(
        "text-generation",
        model=json_model,
        tokenizer=json_tokenizer,
        max_new_tokens=384,
        temperature=0.0,
        do_sample=False,
        pad_token_id=json_tokenizer.pad_token_id,
        return_full_text=False,
        device=pipeline_device,
    )
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    json_llm_chain = prompt | llm
    print("✅ JSON helper ready")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # basic size guard (important)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    print(f"📥 Received image: {file.filename} ({image.size[0]}x{image.size[1]})")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Extract ALL text from this business card. "
                        "Include every name, title, company, phone, email, "
                        "address, and website you can see. Return plain text only."
                    ),
                },
            ],
        },
    ]

    async with request_lock:  # 👈 prevents concurrent CPU meltdown
        print(f"⏳ Processing image...")
        start_time = time.time()

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[1]
        streamer = ProgressStreamer(processor.tokenizer, MAX_NEW_TOKENS, show_progress=False)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                streamer=streamer
            )

        # Only decode the newly generated tokens (exclude prompt)
        generated_tokens = outputs[:, input_len:]
        text = processor.decode(
            generated_tokens[0],
            skip_special_tokens=True
        ).strip()

        elapsed = time.time() - start_time
        print(f"✅ Done in {elapsed:.2f}s")

    # Always run Qwen helper to convert raw text → flat contact JSON
    print("🔄 Running Qwen helper...")
    structured_json, helper_raw = generate_contact_json(text)

    # Save result to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(file.filename or "unknown").stem
    result_file = OUTPUT_DIR / f"{timestamp}_{safe_filename}.json"

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "image_size": f"{image.size[0]}x{image.size[1]}",
        "processing_time_seconds": round(elapsed, 2),
        "raw_text": text,
        "structured_json": structured_json,
        "json_helper_output": helper_raw,
    }

    result_file.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
    print(f"💾 Saved to {result_file}")

    return {
        "raw_text": text,
        "structured_json": structured_json,
        "json_helper_output": helper_raw,
    }
