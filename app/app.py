import os, json, uuid, threading, re, subprocess, time, requests, sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(os.environ.get("STORAGE_DIR", "/tmp/autoclipai"))
UPLOADS  = BASE_DIR / "uploads"
OUTPUTS  = BASE_DIR / "outputs"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

AAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
jobs = {}

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(job_id, url, settings):
    try:
        update(job_id, 5,  "⬇️ Baixando vídeo...",             "download")
        video_path, dl_error = download_video(url, job_id)
        if not video_path:
            return fail(job_id, f"Download falhou: {dl_error}")

        update(job_id, 18, "🔊 Extraindo áudio...",             "download")
        audio_path = extract_audio(video_path, job_id)

        update(job_id, 30, "🎙️ Transcrevendo (AssemblyAI)...", "transcribe")
        transcript, raw_segments = transcribe_assemblyai(audio_path, job_id)
        if not transcript:
            return fail(job_id, "Falha na transcrição. Verifique a chave AssemblyAI.")

        update(job_id, 52, "🧠 Analisando semântica e hooks...", "analyze")
        segments = analyze_transcript(transcript, raw_segments, settings)

        update(job_id, 68, "📈 Calculando Virality Score™...", "score")
        scored = score_clips(segments)

        update(job_id, 78, "✂️ Selecionando melhores clipes...", "score")
        top_clips = pick_top_clips(scored, settings)

        update(job_id, 86, "⚡ Cortando e renderizando...",    "render")
        output_clips = cut_clips(video_path, top_clips, job_id, settings)

        jobs[job_id].update({
            "status": "done", "progress": 100,
            "message": f"✅ {len(output_clips)} clipes gerados!",
            "clips": output_clips,
        })
    except Exception as e:
        fail(job_id, f"{type(e).__name__}: {e}")


def update(job_id, pct, msg, stage):
    jobs[job_id].update({"progress": pct, "message": msg, "stage": stage})

def fail(job_id, msg):
    jobs[job_id].update({"status": "error", "message": f"❌ {msg}"})


# ── Download — usa cliente iOS para bypassar bloqueio do YouTube ──
def download_video(url, job_id):
    out = UPLOADS / job_id
    out.mkdir(exist_ok=True)
    output_tmpl = str(out / "video.%(ext)s")

    # Estratégias em ordem de preferência
    # A chave: extractor-args "youtube:player_client=ios" bypassa o bot-check
    strategies = [
        # 1) Cliente iOS — bypassa "Sign in to confirm you're not a bot"
        ["yt-dlp",
         "--no-playlist", "--no-check-certificates",
         "--extractor-args", "youtube:player_client=ios",
         "-f", "best[height<=480]/best",
         "-o", output_tmpl, url],

        # 2) Cliente mweb (mobile web) — segunda opção
        ["yt-dlp",
         "--no-playlist", "--no-check-certificates",
         "--extractor-args", "youtube:player_client=mweb",
         "-f", "best[height<=480]/best",
         "-o", output_tmpl, url],

        # 3) Cliente tv_embedded — terceira opção
        ["yt-dlp",
         "--no-playlist", "--no-check-certificates",
         "--extractor-args", "youtube:player_client=tv_embedded",
         "-f", "best[height<=480]/best",
         "-o", output_tmpl, url],
    ]

    last_error = ""
    for i, cmd in enumerate(strategies):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        log = result.stdout[-1500:] + "\n---STDERR---\n" + result.stderr[-1500:]
        jobs[job_id][f"dl_log_{i}"] = log

        videos = list(out.glob("video.*"))
        if videos:
            return str(videos[0]), None

        last_error = result.stderr[-600:] if result.stderr else result.stdout[-600:]

    return None, last_error


# ── Audio ─────────────────────────────────────────────────────
def extract_audio(video_path, job_id):
    audio_path = UPLOADS / job_id / "audio.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn", "-b:a", "64k",
        str(audio_path)
    ], capture_output=True, timeout=300)
    return str(audio_path)


# ── AssemblyAI ────────────────────────────────────────────────
def transcribe_assemblyai(audio_path, job_id):
    if not AAI_KEY:
        return None, []
    headers = {"authorization": AAI_KEY}
    with open(audio_path, "rb") as f:
        up = requests.post("https://api.assemblyai.com/v2/upload",
                           headers=headers, data=f, timeout=120)
    audio_url = up.json().get("upload_url")
    if not audio_url:
        return None, []
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={"audio_url": audio_url, "language_detection": True, "auto_chapters": True},
        headers={**headers, "content-type": "application/json"}, timeout=30
    )
    tid = tr.json().get("id")
    if not tid:
        return None, []
    for _ in range(120):
        res = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}",
                           headers=headers, timeout=30).json()
        if res.get("status") == "completed":
            words, chapters, segs = res.get("words",[]), res.get("chapters",[]), []
            if chapters:
                for ch in chapters:
                    segs.append({"start": ch["start"]/1000, "end": ch["end"]/1000,
                                 "text": ch.get("summary", ch.get("headline",""))})
            elif words:
                seg_words, seg_start = [], None
                for w in words:
                    if seg_start is None: seg_start = w["start"]/1000
                    seg_words.append(w["text"])
                    if (w["end"]/1000 - seg_start) >= 30:
                        segs.append({"start": seg_start, "end": w["end"]/1000,
                                     "text": " ".join(seg_words)})
                        seg_words, seg_start = [], None
                if seg_words and seg_start is not None:
                    segs.append({"start": seg_start, "end": words[-1]["end"]/1000,
                                 "text": " ".join(seg_words)})
            jobs[job_id]["raw_segments"] = segs
            return res.get("text",""), segs
        elif res.get("status") == "error":
            return None, []
        time.sleep(5)
    return None, []


# ── Análise & Score ───────────────────────────────────────────
HOOK_PAT = [
    r'\b(segredo|secret|nunca contei|the truth|a verdade|how i|como eu|why i|por que eu)\b',
    r'^["\']', r'\?$',
    r'\b(imagine|picture this|e se)\b',
    r'\b(\d+\s*(ways|tips|dicas|razões|mistakes|erros))\b',
]
FLOW_KW = ['porque','portanto','então','mas','porém','contudo',
           'because','therefore','so','but','however',
           'primeiro','segundo','first','second','third']
VALUE_PAT = [
    r'\b(como fazer|how to|tutorial|dica|tip|aprenda|learn|descubra)\b',
    r'\b(resultado|result|exemplo|example)\b',
    r'\b(solução|solution|resposta|answer|conclusão)\b',
]

def analyze_transcript(transcript, raw_segments, settings):
    target = settings.get("clip_duration", 60)
    min_d, max_d = max(15, target-20), target+20
    if raw_segments:
        candidates, deduped = [], []
        for i in range(len(raw_segments)):
            start = raw_segments[i]["start"]
            for j in range(i+1, len(raw_segments)):
                end, dur = raw_segments[j]["end"], raw_segments[j]["end"]-start
                if dur < min_d: continue
                if dur > max_d: break
                candidates.append({"start":start,"end":end,
                                   "text":" ".join(s["text"] for s in raw_segments[i:j+1]),
                                   "duration":round(dur,1)})
        for c in candidates:
            if not any(c["start"]<d["end"] and c["end"]>d["start"] for d in deduped):
                deduped.append(c)
        if deduped: return deduped
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    clips, cursor, window, wdur = [], 0.0, [], 0.0
    for sent in sentences:
        dur = len(sent.split())/2.5; window.append(sent); wdur += dur
        if wdur >= target-10:
            clips.append({"start":round(cursor,1),"end":round(cursor+wdur,1),
                          "text":" ".join(window),"duration":round(wdur,1)})
            cursor += wdur; window, wdur = [], 0.0
    return clips

def score_clips(segments):
    scored = []
    for seg in segments:
        text  = seg["text"].lower()
        first = re.split(r'[.!?]', text)[0]
        hook  = min(25, sum(8 for p in HOOK_PAT if re.search(p, first, re.I))+(5 if text.strip().endswith("?") else 0))
        flow  = min(25, 10+sum(3 for kw in FLOW_KW if kw in text)+(5 if len(re.split(r'[.!?]+',text))>=3 else 0))
        value = min(25, 10+sum(8 for p in VALUE_PAT if re.search(p, text, re.I)))
        trend = min(25, int(len(set(text.split()))/max(1,len(text.split()))*30))
        total = hook+flow+value+trend
        grade = "🔥 Viral" if total>=75 else "⚡ Alto" if total>=55 else "👍 Médio" if total>=35 else "📉 Baixo"
        scored.append({**seg,"hook":hook,"flow":flow,"value":value,"trend":trend,"score":total,"grade":grade})
    return sorted(scored, key=lambda x: x["score"], reverse=True)

def pick_top_clips(scored, settings):
    n, ms = settings.get("num_clips",5), settings.get("min_score",30)
    return ([c for c in scored if c["score"]>=ms] or scored)[:n]

def cut_clips(video_path, clips, job_id, settings):
    out_dir = OUTPUTS/job_id; out_dir.mkdir(exist_ok=True)
    results, aspect = [], settings.get("aspect","16:9")
    for i, clip in enumerate(clips):
        fname = f"clip_{i+1:02d}_score{clip['score']}.mp4"
        out_path = out_dir/fname
        vf = []
        if aspect=="9:16": vf.append("crop=ih*9/16:ih,scale=1080:1920")
        elif aspect=="1:1": vf.append("crop=ih:ih,scale=1080:1080")
        cmd = ["ffmpeg","-y","-ss",str(clip["start"]),"-i",video_path,
               "-t",str(clip["end"]-clip["start"]),
               "-c:v","libx264","-preset","fast","-crf","23",
               "-c:a","aac","-b:a","128k"]
        if vf: cmd+=["-vf",",".join(vf)]
        cmd.append(str(out_path))
        subprocess.run(cmd, capture_output=True, timeout=300)
        size_mb = round(out_path.stat().st_size/1024/1024,1) if out_path.exists() else 0
        results.append({
            "filename":fname,"url":f"/clip/{job_id}/{fname}",
            "start":clip["start"],"end":clip["end"],
            "duration":round(clip["end"]-clip["start"],1),
            "score":clip["score"],"grade":clip["grade"],
            "hook":clip["hook"],"flow":clip["flow"],
            "value":clip["value"],"trend":clip["trend"],
            "preview_text":clip["text"][:120]+"...","size_mb":size_mb,
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
    if not url:
        return jsonify({"error":"URL obrigatória"}), 400
    if not AAI_KEY:
        return jsonify({"error":"❌ Chave AssemblyAI não configurada. Adicione ASSEMBLYAI_API_KEY nas variáveis do Railway."}), 400
    job_id   = str(uuid.uuid4())[:8]
    settings = {
        "num_clips":     int(data.get("num_clips",5)),
        "clip_duration": int(data.get("clip_duration",60)),
        "min_score":     int(data.get("min_score",30)),
        "aspect":        data.get("aspect","16:9"),
        "captions":      bool(data.get("captions",True)),
    }
    jobs[job_id] = {"id":job_id,"status":"running","progress":0,
                    "message":"Iniciando pipeline...","stage":"init","clips":[]}
    threading.Thread(target=run_pipeline, args=(job_id,url,settings), daemon=True).start()
    return jsonify({"job_id":job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({"error":"Job não encontrado"}), 404
    return jsonify(job)

@app.route("/debug/<job_id>")
def debug(job_id):
    job = jobs.get(job_id, {})
    return jsonify({
        "status":   job.get("status"),
        "message":  job.get("message"),
        "dl_log_0": job.get("dl_log_0","(vazio)"),
        "dl_log_1": job.get("dl_log_1","(vazio)"),
        "dl_log_2": job.get("dl_log_2","(vazio)"),
    })

@app.route("/clip/<job_id>/<filename>")
def serve_clip(job_id, filename):
    return send_from_directory(OUTPUTS/job_id, filename)

@app.route("/health")
def health(): return jsonify({"status":"ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=False)
