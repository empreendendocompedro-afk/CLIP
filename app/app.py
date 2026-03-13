import os, json, uuid, threading, re, subprocess, time, requests, sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(os.environ.get("STORAGE_DIR", "/tmp/autoclipai"))
UPLOADS  = BASE_DIR / "uploads"
OUTPUTS  = BASE_DIR / "outputs"
JOBS_DIR = BASE_DIR / "jobs"
for d in [UPLOADS, OUTPUTS, JOBS_DIR]: d.mkdir(parents=True, exist_ok=True)

AAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

# ─── File-based job store ─────────────────────────────────────
def job_read(jid):
    p = JOBS_DIR / f"{jid}.json"
    if not p.exists(): return None
    with open(p) as f: return json.load(f)

def job_write(jid, data):
    p = JOBS_DIR / f"{jid}.json"
    with open(p, "w") as f: json.dump(data, f)

def job_update(jid, **kw):
    job = job_read(jid) or {}
    job.update(kw)
    job_write(jid, job)

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(job_id, url, settings):
    try:
        job_update(job_id, progress=5, message="⬇️ Baixando vídeo...", stage="download")
        video_path, dl_error = download_video(url, job_id, settings)
        if not video_path:
            return job_update(job_id, status="error", message=f"❌ Download falhou: {dl_error}")

        job_update(job_id, progress=18, message="🔊 Extraindo áudio...", stage="download")
        audio_path = extract_audio(video_path, job_id)
        af = Path(audio_path)
        if not af.exists() or af.stat().st_size < 1000:
            return job_update(job_id, status="error", message="❌ Áudio inválido ou muito curto.")

        job_update(job_id, progress=30, message="🎙️ Transcrevendo (AssemblyAI)...", stage="transcribe",
                   audio_size_kb=round(af.stat().st_size/1024, 1))

        transcript, raw_segments, aai_err = transcribe_assemblyai(audio_path, job_id)
        if not transcript:
            return job_update(job_id, status="error", message=f"❌ Transcrição falhou: {aai_err}")

        job_update(job_id, progress=52, message="🧠 Analisando semântica e hooks...", stage="analyze")
        segments = analyze_transcript(transcript, raw_segments, settings)

        # ── Opus Clip-style scoring ──
        job_update(job_id, progress=68, message="📈 Calculando Virality Score™...", stage="score")
        scored = score_clips_opusclip(segments, transcript)

        job_update(job_id, progress=78, message="✂️ Selecionando melhores clipes...", stage="score")
        top_clips = pick_top_clips(scored, settings)

        job_update(job_id, progress=86, message="⚡ Renderizando em alta qualidade...", stage="render")
        output_clips = cut_clips(video_path, top_clips, job_id, settings)

        job_update(job_id, status="done", progress=100,
                   message=f"✅ {len(output_clips)} clipes gerados!",
                   clips=output_clips)
    except Exception as e:
        job_update(job_id, status="error", message=f"❌ {type(e).__name__}: {e}")


# ── Download ──────────────────────────────────────────────────
def download_video(url, job_id, settings):
    out = UPLOADS / job_id
    out.mkdir(exist_ok=True)
    tmpl = str(out / "video.%(ext)s")
    crf  = settings.get("crf", "20")

    # Escolhe resolução baseada na qualidade escolhida
    if crf == "28":   fmt = "best[height<=720]/best"
    elif crf == "16": fmt = "bestvideo[height<=2160]+bestaudio/best"
    else:             fmt = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"

    strategies = [
        ["yt-dlp","--no-playlist","--no-check-certificates",
         "--extractor-args","youtube:player_client=ios",
         "--merge-output-format","mp4","-f",fmt,"-o",tmpl,url],
        ["yt-dlp","--no-playlist","--no-check-certificates",
         "--extractor-args","youtube:player_client=mweb",
         "--merge-output-format","mp4","-f","best[height<=1080]/best","-o",tmpl,url],
        ["yt-dlp","--no-playlist","--no-check-certificates",
         "--extractor-args","youtube:player_client=tv_embedded",
         "--merge-output-format","mp4","-f","best","-o",tmpl,url],
    ]
    last_err = ""
    for i, cmd in enumerate(strategies):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        job_update(job_id, **{f"dl_log_{i}": (r.stdout+r.stderr)[-2000:]})
        videos = list(out.glob("video.*"))
        if videos: return str(videos[0]), None
        last_err = r.stderr[-600:] if r.stderr else r.stdout[-600:]
    return None, last_err


# ── Audio ─────────────────────────────────────────────────────
def extract_audio(video_path, job_id):
    audio_path = UPLOADS / job_id / "audio.mp3"
    r = subprocess.run([
        "ffmpeg","-y","-i",video_path,
        "-ar","16000","-ac","1","-vn","-b:a","64k",str(audio_path)
    ], capture_output=True, timeout=300)
    log = (r.stdout+r.stderr)
    if isinstance(log, bytes): log = log.decode(errors="ignore")
    job_update(job_id, ffmpeg_log=log[-500:])
    return str(audio_path)


# ── AssemblyAI ────────────────────────────────────────────────
def transcribe_assemblyai(audio_path, job_id):
    if not AAI_KEY: return None, [], "ASSEMBLYAI_API_KEY não definida"
    headers = {"authorization": AAI_KEY}
    try:
        with open(audio_path,"rb") as f:
            up = requests.post("https://api.assemblyai.com/v2/upload",
                               headers=headers, data=f, timeout=120)
        job_update(job_id, aai_upload_status=up.status_code)
        if up.status_code == 401: return None, [], "Chave inválida (401)"
        if up.status_code != 200: return None, [], f"Upload HTTP {up.status_code}"
        audio_url = up.json().get("upload_url")
        if not audio_url: return None, [], "Sem upload_url"
    except Exception as e: return None, [], f"Upload error: {e}"

    try:
        tr = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json={"audio_url": audio_url, "speech_models": ["universal-2"],
                  "language_detection": True, "auto_chapters": True,
                  "sentiment_analysis": True, "auto_highlights": True},
            headers={**headers,"content-type":"application/json"}, timeout=30
        )
        job_update(job_id, aai_transcript_status=tr.status_code,
                   aai_transcript_resp=str(tr.json())[:200])
        tid = tr.json().get("id")
        if not tid: return None, [], f"Sem ID: {tr.json()}"
    except Exception as e: return None, [], f"Transcript request error: {e}"

    for attempt in range(120):
        try:
            res = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}",
                               headers=headers, timeout=30).json()
            st = res.get("status")
            job_update(job_id, aai_poll_status=st, aai_poll_attempt=attempt)
            if st == "completed":
                words    = res.get("words", [])
                chapters = res.get("chapters", [])
                highlights = res.get("auto_highlights_result", {}).get("results", [])
                sentiments = res.get("sentiment_analysis_results", [])
                segs = []
                if chapters:
                    for ch in chapters:
                        segs.append({"start": ch["start"]/1000, "end": ch["end"]/1000,
                                     "text": ch.get("summary", ch.get("headline","")),
                                     "headline": ch.get("headline","")})
                elif words:
                    sw, ss = [], None
                    for w in words:
                        if ss is None: ss = w["start"]/1000
                        sw.append(w["text"])
                        if (w["end"]/1000 - ss) >= 30:
                            segs.append({"start":ss,"end":w["end"]/1000,"text":" ".join(sw),"headline":""})
                            sw, ss = [], None
                    if sw and ss is not None:
                        segs.append({"start":ss,"end":words[-1]["end"]/1000,"text":" ".join(sw),"headline":""})

                # Guardar highlights e sentimentos para o scoring
                job_update(job_id, raw_segments=segs,
                           aai_highlights=[h["text"] for h in highlights[:20]],
                           aai_sentiments=sentiments[:50])
                return res.get("text",""), segs, None
            elif st == "error":
                return None, [], f"AssemblyAI erro: {res.get('error','?')}"
        except Exception as e:
            job_update(job_id, aai_poll_error=str(e))
        time.sleep(5)
    return None, [], "Timeout (10 min)"


# ── Opus Clip–style Scoring ───────────────────────────────────
# Baseado na documentação oficial: Hook + Flow + Value + Trend (0–25 cada = 0–99 total)

HOOK_STRONG = [
    r'\b(você nunca|você sabia|a verdade|segredo|erro que|como eu|por que eu|o que acontece|nunca mais|pare de|nunca faça)\b',
    r'\b(you never|the truth about|secret to|how i|why i|stop doing|never do|biggest mistake)\b',
    r'^\s*["\']',
    r'\b(imagine se|e se eu te dissesse|o que te impede|por que a maioria)\b',
    r'\b(imagine if|what if i told you|what\'s stopping you|why most people)\b',
]
HOOK_MEDIUM = [
    r'\b(\d+\s*(dicas|erros|razões|passos|maneiras|tips|mistakes|reasons|steps|ways))\b',
    r'\b(hoje|agora|urgente|atenção|importante|today|right now|urgent|attention)\b',
    r'\?$',
]

FLOW_POSITIVE = [
    r'\b(porque|portanto|então|assim|logo|ou seja|ou seja|em resumo)\b',
    r'\b(because|therefore|so|thus|hence|in summary|in other words)\b',
    r'\b(primeiro|segundo|terceiro|por último|finalmente|first|second|third|finally|lastly)\b',
    r'\b(por exemplo|como por exemplo|for example|such as|for instance)\b',
]
FLOW_NEGATIVE = [
    r'\b(hm+|uh+|ah+|é+|sabe|tipo|né|bem)\b',  # filler words
    r'\.{3,}',  # ellipsis — pauses
]

VALUE_HIGH = [
    r'\b(como fazer|tutorial|passo a passo|guia|aprenda|how to|step by step|guide|learn)\b',
    r'\b(resultado|prova|funciona|testei|descobri|result|proof|it works|i tested|i found)\b',
    r'\b(economize|ganhe|lucre|mude|transforme|save|earn|profit|change|transform)\b',
]
VALUE_MEDIUM = [
    r'\b(dica|conselho|exemplo|tip|advice|example)\b',
    r'\b(melhor|pior|mais fácil|mais rápido|best|worst|easiest|fastest)\b',
]

TREND_SIGNALS = [
    r'\b(viral|tendência|trend|todo mundo|everyone|everybody)\b',
    r'\b(2024|2025|agora|novo|nova|new|latest|recent)\b',
    r'\b(tiktok|instagram|youtube|reels|shorts|linkedin)\b',
    r'\b(ia|inteligência artificial|ai|chatgpt|gpt|machine learning)\b',
]

def score_clips_opusclip(segments, full_transcript):
    """
    Scoring inspirado no Opus Clip:
    - Hook  (0-25): poder do início — pergunta, afirmação forte, dado surpreendente
    - Flow  (0-25): coerência — transições lógicas, início + meio + fim, sem fillers
    - Value (0-25): valor entregue — ensina, inspira, prova, resolve problema
    - Trend (0-25): alinhamento com tendências — tópicos atuais, plataformas, IA
    Total: 0-99 (como o Opus Clip)
    """
    scored = []
    hl_texts = " ".join(full_transcript.split()[:300]).lower()  # contexto geral

    for seg in segments:
        text  = seg["text"].lower().strip()
        lines = re.split(r'[.!?]+', text)
        first = lines[0] if lines else text
        words = text.split()
        wcount = max(1, len(words))

        # ── HOOK (0–25) ──────────────────────────────────────────
        hook = 8  # base: todo clipe tem algum valor
        for pat in HOOK_STRONG:
            if re.search(pat, first, re.I): hook += 6
        for pat in HOOK_MEDIUM:
            if re.search(pat, first, re.I): hook += 3
        # Bônus: começa direto ao ponto (primeiras 5 palavras já trazem substância)
        if len(first.split()) >= 4: hook += 2
        hook = min(25, hook)

        # ── FLOW (0–25) ──────────────────────────────────────────
        flow = 8
        for pat in FLOW_POSITIVE:
            if re.search(pat, text, re.I): flow += 3
        # Penaliza fillers
        filler_count = sum(1 for pat in FLOW_NEGATIVE if re.search(pat, text, re.I))
        flow -= filler_count * 2
        # Bônus: múltiplas frases (narrativa completa)
        sentence_count = len([l for l in lines if l.strip()])
        if sentence_count >= 3: flow += 4
        if sentence_count >= 5: flow += 2
        # Bônus: duração adequada (nem muito curto nem muito longo)
        dur = seg.get("duration", seg["end"] - seg["start"])
        if 30 <= dur <= 90: flow += 3
        flow = max(0, min(25, flow))

        # ── VALUE (0–25) ─────────────────────────────────────────
        value = 6
        for pat in VALUE_HIGH:
            if re.search(pat, text, re.I): value += 6
        for pat in VALUE_MEDIUM:
            if re.search(pat, text, re.I): value += 3
        # Bônus: densidade de informação (palavras únicas / total)
        unique_ratio = len(set(words)) / wcount
        if unique_ratio > 0.7: value += 3
        elif unique_ratio > 0.5: value += 1
        value = min(25, value)

        # ── TREND (0–25) ─────────────────────────────────────────
        trend = 5
        for pat in TREND_SIGNALS:
            if re.search(pat, text, re.I): trend += 5
        # Bônus: headline do capítulo (AssemblyAI auto-chapters)
        if seg.get("headline"): trend += 3
        trend = min(25, trend)

        total = hook + flow + value + trend  # 0–100 máx
        # Mapeia para 0–99 como o Opus Clip
        score = min(99, total)

        grade = ("🔥 Viral"  if score >= 75 else
                 "⚡ Alto"   if score >= 55 else
                 "👍 Médio"  if score >= 35 else
                 "📉 Baixo")

        scored.append({**seg,
                       "hook": hook, "flow": flow, "value": value, "trend": trend,
                       "score": score, "grade": grade})

    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ── Transcript analysis ───────────────────────────────────────
def analyze_transcript(transcript, raw_segments, settings):
    target = settings.get("clip_duration", 60)
    min_d, max_d = max(15, target-20), target+20

    if raw_segments:
        candidates, deduped = [], []
        for i in range(len(raw_segments)):
            start = raw_segments[i]["start"]
            for j in range(i+1, len(raw_segments)):
                end = raw_segments[j]["end"]; dur = end-start
                if dur < min_d: continue
                if dur > max_d: break
                text = " ".join(s["text"] for s in raw_segments[i:j+1])
                headline = raw_segments[i].get("headline","")
                candidates.append({"start":start,"end":end,"text":text,
                                    "duration":round(dur,1),"headline":headline})
        for c in candidates:
            if not any(c["start"]<d["end"] and c["end"]>d["start"] for d in deduped):
                deduped.append(c)
        if deduped: return deduped

    # Fallback
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    clips, cursor, window, wdur = [], 0.0, [], 0.0
    for sent in sentences:
        dur = len(sent.split())/2.5; window.append(sent); wdur += dur
        if wdur >= target-10:
            clips.append({"start":round(cursor,1),"end":round(cursor+wdur,1),
                          "text":" ".join(window),"duration":round(wdur,1),"headline":""})
            cursor += wdur; window, wdur = [], 0.0
    return clips


def pick_top_clips(scored, settings):
    n, ms = settings.get("num_clips",5), settings.get("min_score",30)
    return ([c for c in scored if c["score"]>=ms] or scored)[:n]


# ── Cut clips — qualidade melhorada ──────────────────────────
def cut_clips(video_path, clips, job_id, settings):
    out_dir = OUTPUTS/job_id; out_dir.mkdir(exist_ok=True)
    results, aspect = [], settings.get("aspect","9:16")
    crf = str(settings.get("crf", "20"))

    # Detectar resolução do vídeo original
    probe = subprocess.run([
        "ffprobe","-v","quiet","-print_format","json",
        "-show_streams",video_path
    ], capture_output=True, text=True)
    try:
        streams = json.loads(probe.stdout).get("streams",[])
        vs = next((s for s in streams if s.get("codec_type")=="video"), {})
        orig_w = int(vs.get("width",1920))
        orig_h = int(vs.get("height",1080))
    except:
        orig_w, orig_h = 1920, 1080

    for i, clip in enumerate(clips):
        fname    = f"clip_{i+1:02d}_score{clip['score']}.mp4"
        out_path = out_dir/fname

        vf = []
        if aspect == "9:16":
            # Crop para vertical mantendo a melhor parte central
            crop_w = min(orig_w, int(orig_h * 9/16))
            crop_h = orig_h
            vf.append(f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:0")
            vf.append("scale=1080:1920:flags=lanczos")
        elif aspect == "1:1":
            sq = min(orig_w, orig_h)
            vf.append(f"crop={sq}:{sq}:(iw-{sq})/2:(ih-{sq})/2")
            vf.append("scale=1080:1080:flags=lanczos")
        else:  # 16:9
            vf.append("scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease")
            vf.append("pad=1920:1080:-1:-1:color=black")

        # Remover silêncios se ativado
        if settings.get("remove_silence", True):
            vf.append("silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip["start"]),
            "-i", video_path,
            "-t", str(clip["end"] - clip["start"]),
            # Vídeo — qualidade alta
            "-c:v", "libx264",
            "-preset", "slow",      # melhor compressão (mais lento mas maior qualidade)
            "-crf", crf,            # 16=ultra, 20=HD, 28=rápido
            "-profile:v", "high",
            "-level", "4.1",
            "-pix_fmt", "yuv420p",
            # Áudio — qualidade alta
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
        ]
        if vf: cmd += ["-vf", ",".join(vf)]
        cmd += ["-movflags", "+faststart", str(out_path)]

        subprocess.run(cmd, capture_output=True, timeout=600)

        size_mb = round(out_path.stat().st_size/1024/1024,1) if out_path.exists() else 0
        results.append({
            "filename": fname, "url": f"/clip/{job_id}/{fname}",
            "start": clip["start"], "end": clip["end"],
            "duration": round(clip["end"]-clip["start"],1),
            "score": clip["score"], "grade": clip["grade"],
            "hook": clip["hook"], "flow": clip["flow"],
            "value": clip["value"], "trend": clip["trend"],
            "preview_text": clip["text"][:140]+"...",
            "size_mb": size_mb,
        })
    return results


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index(): return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    url  = data.get("url","").strip()
    if not url: return jsonify({"error":"URL obrigatória"}), 400
    if not AAI_KEY:
        return jsonify({"error":"❌ Chave AssemblyAI não configurada."}), 400
    job_id   = str(uuid.uuid4())[:8]
    settings = {
        "num_clips":      int(data.get("num_clips",5)),
        "clip_duration":  int(data.get("clip_duration",60)),
        "min_score":      int(data.get("min_score",30)),
        "aspect":         data.get("aspect","9:16"),
        "crf":            str(data.get("crf","20")),
        "captions":       bool(data.get("captions",True)),
        "remove_silence": bool(data.get("remove_silence",True)),
    }
    job_write(job_id, {"id":job_id,"status":"running","progress":0,
                       "message":"Iniciando pipeline...","stage":"init","clips":[]})
    threading.Thread(target=run_pipeline, args=(job_id,url,settings), daemon=True).start()
    return jsonify({"job_id":job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = job_read(job_id)
    if not job: return jsonify({"error":"Job não encontrado"}), 404
    return jsonify(job)

@app.route("/debug/<job_id>")
def debug(job_id):
    job = job_read(job_id)
    if not job: return jsonify({"error":"Job não encontrado"}), 404
    return jsonify({k:v for k,v in job.items() if k not in ["clips","raw_segments"]})

@app.route("/test-aai")
def test_aai():
    if not AAI_KEY: return jsonify({"ok":False,"erro":"Chave não definida"})
    try:
        r = requests.get("https://api.assemblyai.com/v2/transcript",
                         headers={"authorization":AAI_KEY}, timeout=10)
        if r.status_code == 200: return jsonify({"ok":True,"msg":"Chave válida!"})
        elif r.status_code == 401: return jsonify({"ok":False,"erro":"Chave INVÁLIDA — 401"})
        else: return jsonify({"ok":False,"erro":f"HTTP {r.status_code}"})
    except Exception as e: return jsonify({"ok":False,"erro":str(e)})

@app.route("/clip/<job_id>/<filename>")
def serve_clip(job_id, filename):
    return send_from_directory(OUTPUTS/job_id, filename)

@app.route("/health")
def health(): return jsonify({"status":"ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=False)
