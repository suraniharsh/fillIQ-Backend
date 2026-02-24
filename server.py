import io
import sys
import asyncio
import time
import json
import base64
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import HuggingFacePipeline
from pydantic import BaseModel, Field
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
JSON_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MAX_NEW_TOKENS = 512

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
json_chain = None
json_format_instructions = None
json_parser = None
json_prompt = None


class ContactBlock(BaseModel):
    name: str | None = Field(None, description="Full name on the card")
    role: str | None = Field(None, description="Job title or role")
    company: str | None = Field(None, description="Company or organization name")


class CommunicationBlock(BaseModel):
    phones: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    website: str | None = None


class LocationBlock(BaseModel):
    address: str | None = None


class BrandingBlock(BaseModel):
    logo_text: str | None = None
    brand_notes: str | None = None


class ConfidenceBlock(BaseModel):
    score: float | None = Field(None, description="Confidence score between 0 and 1")
    notes: str | None = None


class ContactPayload(BaseModel):
    contact: ContactBlock
    communications: CommunicationBlock
    location: LocationBlock
    branding: BrandingBlock
    confidence: ConfidenceBlock

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


def extract_json_block(text: str) -> dict | None:
    """Grab the first JSON object embedded in a string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def generate_contact_json(raw_text: str) -> tuple[dict | None, str | None]:
    """Use LangChain pipeline to transform OCR text into structured JSON."""

    if json_chain is None or json_parser is None or json_format_instructions is None:
        return None, None

    try:
        structured = json_chain.invoke(
            {
                "raw_text": raw_text,
                "format_instructions": json_format_instructions,
            }
        )

        if isinstance(structured, ContactPayload):
            structured_dict = structured.dict()
        elif isinstance(structured, dict):
            structured_dict = structured
        else:
            return None, None

        return structured_dict, json.dumps(structured_dict, ensure_ascii=False)
    except OutputParserException:
        return None, None
    except Exception:
        return None, None


@app.on_event("startup")
async def load_model():
    global processor, model, json_tokenizer, json_model, json_chain, json_format_instructions, json_parser, json_prompt

    print("🔹 Loading processor...")
    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)

    print("🔹 Loading model (CPU)... this will take time")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    model.eval()
    print("✅ Model ready")

    print(f"🔹 Loading JSON helper model ({JSON_MODEL_ID})...")
    json_tokenizer = AutoTokenizer.from_pretrained(JSON_MODEL_ID)
    if json_tokenizer.pad_token is None:
        json_tokenizer.pad_token = json_tokenizer.eos_token
    json_model = AutoModelForCausalLM.from_pretrained(
        JSON_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    json_model.eval()

    parser = PydanticOutputParser(pydantic_object=ContactPayload)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert data extraction assistant. Convert business-card OCR text into the requested JSON schema. Always reply with valid JSON only.",
            ),
            (
                "user",
                "Extract contact information from the following OCR text.\n\nRAW OCR INPUT:\n{raw_text}\n\nFollow this JSON schema exactly:\n{format_instructions}",
            ),
        ]
    )

    generation_pipeline = hf_pipeline(
        "text-generation",
        model=json_model,
        tokenizer=json_tokenizer,
        max_new_tokens=384,
        temperature=0.0,
        do_sample=False,
        pad_token_id=json_tokenizer.pad_token_id,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    json_chain = prompt | llm | parser
    json_parser = parser
    json_format_instructions = parser.get_format_instructions()
    json_prompt = prompt
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

    # Convert image to base64 data URI
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"

    # Prompt chain to encourage structured JSON output
    prompt_chain = [
        "Please extract the text from the image, identifying labels and values for name, company, mobile number, email, address, role, and any logos or brand signals. Return only a JSON object with keys that match these pieces of information (use snake_case if unsure).",
        "Double-check the extracted information and respond with a clean JSON payload that includes: contact (name, company, role), communications (phones, emails), location (address), branding (logo_text, brand_notes), and confidence hints. If no data exists for a field, set it to null or an empty array. Do not prepend or append any prose."
    ]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_chain[0]},
                {"type": "image", "url": image_uri},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_chain[1]},
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

    # Save result to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(file.filename or "unknown").stem
    result_file = OUTPUT_DIR / f"{timestamp}_{safe_filename}.json"
    
    # Try to read JSON directly from model output, otherwise synthesize via helper LLM
    structured_json = extract_json_block(text)
    helper_raw_json = None
    if structured_json is None:
        structured_json, helper_raw_json = generate_contact_json(text)
    else:
        helper_raw_json = json.dumps(structured_json)

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "image_size": f"{image.size[0]}x{image.size[1]}",
        "processing_time_seconds": round(elapsed, 2),
        "raw_text": text,
        "structured_json": structured_json,
        "json_helper_output": helper_raw_json,
    }
    
    result_file.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
    print(f"💾 Saved to {result_file}")

    return {
        "raw_text": text,
        "structured_json": structured_json,
        "json_helper_output": helper_raw_json,
    }
