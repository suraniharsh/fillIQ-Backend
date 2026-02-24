"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, FileText, Loader2, Repeat, Sparkles } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type UploadStatus = "idle" | "loading" | "success" | "error";

export default function Page() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [result, setResult] = useState<string>("");
  const [preview, setPreview] = useState<string | null>(null);
  const [message, setMessage] = useState("No card yet");

  const handleFile = async (file: File) => {
    if (!file) return;
    setStatus("loading");
    setPreview(URL.createObjectURL(file));
    setMessage("Scanning card...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/ocr`, {
        method: "POST",
        body: formData
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload?.error ?? "Server error");
      }

      // The API now returns { raw_text, structured_json, json_helper_output }
      const structuredJson = payload?.structured_json;
      const helperFallback = payload?.json_helper_output;
      const displayJson = structuredJson
        ? JSON.stringify(structuredJson, null, 2)
        : helperFallback ?? JSON.stringify(payload, null, 2);

      setResult(displayJson);
      setMessage(structuredJson ? "Contact extracted" : "Result ready");
      setStatus("success");
    } catch (error) {
      console.error(error);
      const errorMessage = error instanceof Error ? error.message : "Something went wrong";
      setMessage(errorMessage);
      setStatus("error");
    }
  };

  const handleDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      await handleFile(file);
    }
  };

  const handleSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      void handleFile(file);
    }
    event.target.value = "";
  };

  const handleCameraClick = () => {
    cameraInputRef.current?.click();
  };

  const handleClick = () => fileInputRef.current?.click();

  return (
    <main className="flex min-h-screen w-full flex-col items-center justify-center px-4 py-12">
      <section className="w-full max-w-5xl space-y-8 rounded-[40px] border border-white/10 bg-white/[.04] p-6 shadow-[0_40px_80px_rgba(5,6,10,0.6)] backdrop-blur-2xl">
        <div className="flex items-center justify-between gap-2 text-xs uppercase tracking-[0.4em] text-white/50">
          <Sparkles className="h-5 w-5" />
          <span>Lite scan</span>
          <span className="text-ember">Live</span>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={(event) => event.preventDefault()}
          className="relative flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-white/30 bg-white/5 p-8 text-center transition hover:border-ember/60"
          onClick={handleClick}
        >
          <div className="flex h-32 w-32 items-center justify-center rounded-2xl bg-gradient-to-br from-ember/40 to-white/20">
            <Camera className="h-8 w-8 text-white" />
          </div>
          <p className="mt-4 text-sm text-white/80">Tap or drop a card</p>
          <p className="text-xs text-white/40">Auto upload · No text fields</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="absolute inset-0 h-full w-full opacity-0"
            onChange={handleSelect}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleSelect}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm uppercase tracking-[0.3em] text-white/60">
            <span className="h-3 w-3 rounded-full bg-ember/80" />
            <span>{status === "loading" ? "scanning" : message}</span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleClick}>
              <Repeat className="h-4 w-4" />
              <span className="ml-2 text-[0.65rem]">Rescan</span>
            </Button>
            <Button disabled={status === "loading"} onClick={handleClick}>
              <FileText className="h-4 w-4" />
              <span className="ml-2 text-[0.65rem]">Upload</span>
            </Button>
            <Button variant="ghost" size="sm" onClick={handleCameraClick}>
              <Camera className="h-4 w-4" />
              <span className="ml-2 text-[0.65rem]">Capture</span>
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="h-72 overflow-hidden rounded-3xl">
            <div className="h-full w-full rounded-3xl bg-gradient-to-br from-white/10 to-white/0">
              {preview ? (
                <img
                  src={preview}
                  alt="preview"
                  className="h-full w-full object-cover"
                  onLoad={() => URL.revokeObjectURL(preview)}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2">
                  <Loader2 className="h-6 w-6 animate-spin text-white" />
                  <p className="text-xs uppercase tracking-[0.4em] text-white/40">waiting</p>
                </div>
              )}
            </div>
          </Card>
          <Card className="h-72 rounded-3xl border border-white/10 bg-[#03040a]/80 p-4 shadow-[0_20px_50px_rgba(5,6,10,0.4)]">
            <div className="flex items-center gap-3 pb-2">
              <span className="text-[0.65rem] uppercase tracking-[0.4em] text-white/40">Contact JSON</span>
              <div
                className={`h-2 w-2 rounded-full ${status === "success" ? "bg-emerald-500" : status === "error" ? "bg-red-500" : "bg-white/40"}`}
              />
            </div>
            <div className="h-[calc(100%-2rem)] overflow-y-auto text-white/90">
              <pre className="whitespace-pre-wrap text-xs font-mono leading-5">{result || "{\n  \"contact\": { ... },\n  \"communications\": { ... },\n  \"location\": { ... }\n}"}</pre>
            </div>
          </Card>
        </div>
      </section>
    </main>
  );
}
