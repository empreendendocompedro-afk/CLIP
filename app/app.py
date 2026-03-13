import os, json, uuid, threading, re, subprocess, time, requests, sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(os.environ.get("STORAGE_DIR", "/tmp/autoclipai"))
UPLOADS  = BASE_DIR / "uploads"
OUTPUTS  = BASE_DIR / "outputs"
JOBS_DIR = BASE_DIR / "jobs"
for d in [UPLOADS, OUTPUTS, JOBS_DIR]: d.mkdir(parents=True, exist_ok=True)

# ── Cookies do YouTube ────────────────────────────────────────
COOKIES_FILE = BASE_DIR / "yt_cookies.txt"

def setup_cookies():
    """Grava a variável YOUTUBE_COOKIES em arquivo para o yt-dlp usar."""
    cookie_env = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if cookie_env:
        with open(COOKIES_FILE, "w", encoding="utf-8") as f:
            f.write(cookie_env)
        return True
    return False

HAS_COOKIES = setup_cookies()


AAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
MAX_DURATION_SEC = 3600  # 60 minutos

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

# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(job_id, url, settings):
    try:
        # 1. Validar duração
        job_update(job_id, progress=2, message="🔍 Verificando vídeo...", stage="download")
        duration_sec, dur_err = get_video_duration(url)
        if duration_sec and duration_sec > MAX_DURATION_SEC:
            mins = duration_sec // 60
            return job_update(job_id, status="error",
                message=f"❌ Vídeo muito longo ({mins} min). O limite é 60 minutos.")
        mins_label = f" ({int(duration_sec//60)} min)" if duration_sec else ""
        job_update(job_id, progress=5, message=f"⬇️ Baixando vídeo{mins_label}...", stage="download")

        # 2. Download
        video_path, dl_error = download_video(url, job_id, settings)
        if not video_path:
            return job_update(job_id, status="error", message=f"❌ Download falhou: {dl_error}")

        # 3. Áudio
        job_update(job_id, progress=18, message="🔊 Extraindo áudio...", stage="download")
        audio_path = extract_audio(video_path, job_id)
        af = Path(audio_path)
        if not af.exists() or af.stat().st_size < 1000:
            return job_update(job_id, status="error", message="❌ Áudio inválido ou muito curto.")

        # 4. Transcrição
        job_update(job_id, progress=30,
            message="🎙️ Transcrevendo (AssemblyAI)...", stage="transcribe",
            audio_size_kb=round(af.stat().st_size/1024, 1))
        transcript, raw_segments, words_data, aai_err = transcribe_assemblyai(audio_path, job_id)
        if not transcript:
            return job_update(job_id, status="error", message=f"❌ Transcrição falhou: {aai_err}")

        # 5. Análise
        job_update(job_id, progress=52, message="🧠 Analisando semântica e hooks...", stage="analyze")
        segments = analyze_transcript(transcript, raw_segments, settings)

        # 6. Score
        job_update(job_id, progress=68, message="📈 Calculando Virality Score™...", stage="score",
                   segments_count=len(segments))
        if not segments:
            return job_update(job_id, status="error",
                message="❌ Nenhum segmento identificado no vídeo. O vídeo tem fala/diálogo?")
        scored = score_clips_opusclip(segments, transcript)
        job_update(job_id, scored_count=len(scored),
                   top_score=scored[0]["score"] if scored else 0)

        # 7. Selecionar
        job_update(job_id, progress=78, message="✂️ Selecionando melhores clipes...", stage="score")
        top_clips = pick_top_clips(scored, settings)

        # 8. Render com legendas + sem silêncios
        job_update(job_id, progress=86, message="⚡ Renderizando clipes com legendas...", stage="render")
        output_clips = cut_clips(video_path, top_clips, job_id, settings, words_data)

        if not output_clips:
            top = scored[0]["score"] if scored else 0
            return job_update(job_id, status="error",
                message=f"❌ Render falhou em todos os clipes. Score máximo: {top}. "
                        "Verifique o debug para mais detalhes.")
        job_update(job_id, status="done", progress=100,
                   message=f"✅ {len(output_clips)} clipes gerados!",
                   clips=output_clips)
    except Exception as e:
        job_update(job_id, status="error", message=f"❌ {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# VALIDAÇÃO DE DURAÇÃO
# ═══════════════════════════════════════════════════════════════

def get_video_duration(url):
    try:
        cmd = ["yt-dlp","--no-playlist","--dump-json","--no-check-certificates"]
        if COOKIES_FILE.exists():
            cmd += ["--cookies", str(COOKIES_FILE)]
        cmd.append(url)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return None, r.stderr[:200]
        info = json.loads(r.stdout.split("\n")[0])
        return int(info.get("duration", 0)), None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════


def download_video(url, job_id, settings):
    out  = UPLOADS / job_id
    out.mkdir(exist_ok=True)
    tmpl = str(out / "video.%(ext)s")
    crf  = settings.get("crf", "20")

    if crf == "28":   fmt = "best[height<=720]/best"
    elif crf == "16": fmt = "bestvideo[height<=2160]+bestaudio/best"
    else:             fmt = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"

    # Base de argumentos comuns
    base = ["yt-dlp", "--no-playlist", "--no-check-certificates",
            "--merge-output-format", "mp4", "-o", tmpl]

    # Adiciona cookies se disponível
    if COOKIES_FILE.exists():
        base += ["--cookies", str(COOKIES_FILE)]

    strategies = [
        # 1) Com cookies + cliente iOS (mais compatível com servidor)
        base + ["--extractor-args", "youtube:player_client=ios", "-f", fmt, url],
        # 2) Com cookies + cliente web padrão
        base + ["--extractor-args", "youtube:player_client=web", "-f", fmt, url],
        # 3) Com cookies + melhor disponível
        base + ["-f", "best[height<=1080]/best", url],
    ]

    last_err = ""
    for i, cmd in enumerate(strategies):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        job_update(job_id, **{f"dl_log_{i}": (r.stdout+r.stderr)[-2000:]})
        videos = list(out.glob("video.*"))
        if videos: return str(videos[0]), None
        last_err = r.stderr[-600:] if r.stderr else r.stdout[-600:]
    return None, last_err


# ═══════════════════════════════════════════════════════════════
# ÁUDIO
# ═══════════════════════════════════════════════════════════════

def extract_audio(video_path, job_id):
    audio_path = UPLOADS / job_id / "audio.mp3"
    r = subprocess.run([
        "ffmpeg","-y","-i",video_path,
        "-ar","16000","-ac","1","-vn","-b:a","32k",
        "-t","3600", str(audio_path)
    ], capture_output=True, timeout=300)
    log = r.stdout + r.stderr
    if isinstance(log, bytes): log = log.decode(errors="ignore")
    job_update(job_id, ffmpeg_log=log[-500:])
    return str(audio_path)


# ═══════════════════════════════════════════════════════════════
# ASSEMBLYAI
# ═══════════════════════════════════════════════════════════════

def transcribe_assemblyai(audio_path, job_id):
    if not AAI_KEY: return None, [], [], "ASSEMBLYAI_API_KEY não definida"
    headers = {"authorization": AAI_KEY}

    try:
        with open(audio_path,"rb") as f:
            up = requests.post("https://api.assemblyai.com/v2/upload",
                               headers=headers, data=f, timeout=120)
        job_update(job_id, aai_upload_status=up.status_code)
        if up.status_code == 401: return None, [], [], "Chave inválida (401)"
        if up.status_code != 200: return None, [], [], f"Upload HTTP {up.status_code}"
        audio_url = up.json().get("upload_url")
        if not audio_url: return None, [], [], "Sem upload_url"
    except Exception as e: return None, [], [], f"Erro no upload: {e}"

    try:
        tr = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json={"audio_url": audio_url, "speech_models": ["universal-2"],
                  "language_detection": True, "auto_chapters": True,
                  "sentiment_analysis": True, "auto_highlights": True},
            headers={**headers,"content-type":"application/json"}, timeout=30)
        job_update(job_id, aai_transcript_status=tr.status_code,
                   aai_transcript_resp=str(tr.json())[:200])
        tid = tr.json().get("id")
        if not tid: return None, [], [], f"Sem ID: {tr.json()}"
    except Exception as e: return None, [], [], f"Erro ao solicitar transcrição: {e}"

    for attempt in range(240):
        try:
            res = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}",
                               headers=headers, timeout=30).json()
            st = res.get("status")
            if attempt % 10 == 0 and attempt > 0:
                mins = (attempt*5)//60; secs = (attempt*5)%60
                job_update(job_id,
                    message=f"🎙️ Transcrevendo... {mins}m{secs:02d}s — aguardando AssemblyAI",
                    aai_poll_status=st, aai_poll_attempt=attempt)
            else:
                job_update(job_id, aai_poll_status=st, aai_poll_attempt=attempt)

            if st == "completed":
                words    = res.get("words", [])
                chapters = res.get("chapters", [])
                segs     = []
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
                            segs.append({"start":ss,"end":w["end"]/1000,
                                         "text":" ".join(sw),"headline":""})
                            sw, ss = [], None
                    if sw and ss is not None:
                        segs.append({"start":ss,"end":words[-1]["end"]/1000,
                                     "text":" ".join(sw),"headline":""})
                job_update(job_id, raw_segments=segs)
                return res.get("text",""), segs, words, None
            elif st == "error":
                return None, [], [], f"AssemblyAI erro: {res.get('error','?')}"
        except Exception as e:
            job_update(job_id, aai_poll_error=str(e))
        time.sleep(5)
    return None, [], [], "Timeout (20 min) — tente um vídeo menor que 60 minutos."


# ═══════════════════════════════════════════════════════════════
# GERAÇÃO DE LEGENDAS SRT
# ═══════════════════════════════════════════════════════════════

def seconds_to_srt(s):
    h  = int(s//3600); m = int((s%3600)//60)
    sc = int(s%60);   ms = int(round((s-int(s))*1000))
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def generate_srt(words_data, clip_start, clip_end, srt_path, max_words=8, max_chars=42):
    clip_words = [w for w in words_data
                  if w.get("start") is not None and w.get("end") is not None
                  and (w["start"]/1000) >= clip_start
                  and (w["end"]/1000)   <= clip_end + 0.5]
    if not clip_words: return False

    lines, buf, buf_s, buf_e = [], [], None, None

    def flush():
        if buf and buf_s is not None:
            lines.append((buf_s, buf_e, " ".join(buf)))

    for w in clip_words:
        ws = w["start"]/1000 - clip_start
        we = w["end"]/1000   - clip_start
        text = w.get("text","")
        if buf_s is None: buf_s = max(0.0, ws)
        buf.append(text); buf_e = we
        if len(buf) >= max_words or len(" ".join(buf)) >= max_chars:
            flush(); buf=[]; buf_s=None; buf_e=None

    flush()
    if not lines: return False

    content = ""
    for idx, (s, e, text) in enumerate(lines, 1):
        if e - s < 0.5: e = s + 0.5
        content += f"{idx}\n{seconds_to_srt(s)} --> {seconds_to_srt(e)}\n{text.upper()}\n\n"

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


# ═══════════════════════════════════════════════════════════════
# REMOÇÃO INTELIGENTE DE SILÊNCIOS
# ═══════════════════════════════════════════════════════════════

def smart_silence_removal(input_path, output_path, job_id, clip_index):
    try:
        # Detectar silêncios
        probe = subprocess.run([
            "ffmpeg","-y","-i",input_path,
            "-af","silencedetect=noise=-40dB:d=0.4",
            "-f","null","-"
        ], capture_output=True, text=True, timeout=120)

        log = probe.stderr
        silence_starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
        silence_ends   = [float(m) for m in re.findall(r"silence_end: ([\d.]+)",   log)]

        if not silence_starts or not silence_ends:
            return input_path  # sem silêncio, retorna original

        # Duração total
        dur_probe = subprocess.run([
            "ffprobe","-v","quiet","-print_format","json","-show_format",input_path
        ], capture_output=True, text=True, timeout=30)
        total_dur = float(json.loads(dur_probe.stdout).get("format",{}).get("duration",0))
        if total_dur == 0: return input_path

        # Construir segmentos de fala
        crossfade = 0.05
        speech_segs = []
        cursor = 0.0

        for ss, se in zip(silence_starts, silence_ends):
            seg_s = cursor
            seg_e = max(cursor, ss - crossfade)
            if seg_e - seg_s > 0.15:
                speech_segs.append((seg_s, seg_e))
            cursor = se + crossfade

        if cursor < total_dur - 0.15:
            speech_segs.append((cursor, total_dur))

        if len(speech_segs) <= 1:
            return input_path

        # Cortar segmentos individuais
        tmp_dir   = Path(input_path).parent / f"tmp_{clip_index}"
        tmp_dir.mkdir(exist_ok=True)
        seg_files = []

        for si, (s, e) in enumerate(speech_segs):
            seg_out = tmp_dir / f"s{si:03d}.mp4"
            dur = e - s
            if dur < 0.1: continue
            subprocess.run([
                "ffmpeg","-y","-ss",str(s),"-i",input_path,"-t",str(dur),
                "-c:v","libx264","-preset","fast","-crf","20",
                "-c:a","aac","-b:a","192k",
                "-avoid_negative_ts","make_zero", str(seg_out)
            ], capture_output=True, timeout=120)
            if seg_out.exists() and seg_out.stat().st_size > 500:
                seg_files.append(str(seg_out))

        if len(seg_files) < 2:
            return input_path

        # Concatenar com xfade
        n = len(seg_files)
        inputs = []
        for sf in seg_files: inputs += ["-i", sf]

        fc, vchain, achain = [], "[0:v]", "[0:a]"
        for i in range(1, n):
            vout = "vout" if i == n-1 else f"xv{i}"
            aout = "aout" if i == n-1 else f"xa{i}"
            fc.append(f"{vchain}[{i}:v]xfade=transition=fade:duration={crossfade}:offset=0[{vout}]")
            fc.append(f"{achain}[{i}:a]acrossfade=d={crossfade}[{aout}]")
            vchain = f"[{vout}]"; achain = f"[{aout}]"

        cmd = (["ffmpeg","-y"] + inputs + [
            "-filter_complex", ";".join(fc),
            "-map","[vout]","-map","[aout]",
            "-c:v","libx264","-preset","fast","-crf","20",
            "-c:a","aac","-b:a","192k",
            "-movflags","+faststart", str(output_path)
        ])
        subprocess.run(cmd, capture_output=True, timeout=300)

        for sf in seg_files:
            try: Path(sf).unlink()
            except: pass
        try: tmp_dir.rmdir()
        except: pass

        if output_path.exists() and output_path.stat().st_size > 10000:
            return str(output_path)
        return input_path

    except Exception:
        return input_path  # fallback silencioso


# ═══════════════════════════════════════════════════════════════
# SCORING — Opus Clip style
# ═══════════════════════════════════════════════════════════════

HOOK_STRONG = [
    r'\b(você nunca|você sabia|a verdade|segredo|erro que|como eu|por que eu|nunca mais|pare de|nunca faça)\b',
    r'\b(you never|the truth about|secret to|how i|why i|stop doing|never do|biggest mistake)\b',
    r'^\s*["\']',
    r'\b(imagine se|e se eu te dissesse|por que a maioria)\b',
    r'\b(imagine if|what if i told you|why most people)\b',
]
HOOK_MEDIUM = [
    r'\b(\d+\s*(dicas|erros|razões|passos|maneiras|tips|mistakes|reasons|steps|ways))\b',
    r'\b(hoje|agora|urgente|importante|today|right now|urgent|attention)\b',
    r'\?$',
]
FLOW_POSITIVE = [
    r'\b(porque|portanto|então|assim|logo|ou seja|em resumo)\b',
    r'\b(because|therefore|so|thus|hence|in summary|in other words)\b',
    r'\b(primeiro|segundo|terceiro|por último|finalmente|first|second|third|finally|lastly)\b',
    r'\b(por exemplo|for example|such as|for instance)\b',
]
FLOW_NEGATIVE = [r'\b(hm+|uh+|ah+|né|sabe|tipo|bem)\b', r'\.{3,}']
VALUE_HIGH    = [
    r'\b(como fazer|tutorial|passo a passo|guia|aprenda|how to|step by step|guide|learn)\b',
    r'\b(resultado|prova|funciona|testei|descobri|result|proof|it works|i tested|i found)\b',
    r'\b(economize|ganhe|lucre|mude|transforme|save|earn|profit|change|transform)\b',
]
VALUE_MEDIUM  = [
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
    scored = []
    for seg in segments:
        text  = seg["text"].lower().strip()
        lines = re.split(r'[.!?]+', text)
        first = lines[0] if lines else text
        words = text.split(); wcount = max(1, len(words))

        hook = 8
        for p in HOOK_STRONG:
            if re.search(p, first, re.I): hook += 6
        for p in HOOK_MEDIUM:
            if re.search(p, first, re.I): hook += 3
        if len(first.split()) >= 4: hook += 2
        hook = min(25, hook)

        flow = 8
        for p in FLOW_POSITIVE:
            if re.search(p, text, re.I): flow += 3
        flow -= sum(1 for p in FLOW_NEGATIVE if re.search(p, text, re.I)) * 2
        sc = len([l for l in lines if l.strip()])
        if sc >= 3: flow += 4
        if sc >= 5: flow += 2
        dur = seg.get("duration", seg["end"]-seg["start"])
        if 30 <= dur <= 90: flow += 3
        flow = max(0, min(25, flow))

        value = 6
        for p in VALUE_HIGH:
            if re.search(p, text, re.I): value += 6
        for p in VALUE_MEDIUM:
            if re.search(p, text, re.I): value += 3
        ur = len(set(words))/wcount
        if ur > 0.7: value += 3
        elif ur > 0.5: value += 1
        value = min(25, value)

        trend = 5
        for p in TREND_SIGNALS:
            if re.search(p, text, re.I): trend += 5
        if seg.get("headline"): trend += 3
        trend = min(25, trend)

        score = min(99, hook+flow+value+trend)
        grade = ("🔥 Viral" if score>=75 else "⚡ Alto" if score>=55
                 else "👍 Médio" if score>=35 else "📉 Baixo")
        scored.append({**seg,"hook":hook,"flow":flow,"value":value,
                       "trend":trend,"score":score,"grade":grade})
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def analyze_transcript(transcript, raw_segments, settings):
    target = settings.get("clip_duration", 60)
    min_d  = max(15, target - 20)
    max_d  = target + 20

    # Tentativa 1: usar capítulos/segmentos da AssemblyAI
    if raw_segments:
        candidates, deduped = [], []
        for i in range(len(raw_segments)):
            start = raw_segments[i]["start"]
            for j in range(i+1, len(raw_segments)):
                end = raw_segments[j]["end"]; dur = end - start
                if dur < min_d: continue
                if dur > max_d: break
                candidates.append({
                    "start":    start, "end": end,
                    "text":     " ".join(s["text"] for s in raw_segments[i:j+1]),
                    "duration": round(dur, 1),
                    "headline": raw_segments[i].get("headline", ""),
                })
        for c in candidates:
            if not any(c["start"] < d["end"] and c["end"] > d["start"] for d in deduped):
                deduped.append(c)
        if deduped:
            return deduped

        # Tentativa 1b: segmentos individuais com duração suficiente (janela mais larga)
        deduped2 = []
        for i in range(len(raw_segments)):
            for j in range(i, len(raw_segments)):
                end = raw_segments[j]["end"]
                dur = end - raw_segments[i]["start"]
                if dur < 20: continue   # mínimo absoluto de 20s
                if dur > target + 40: break
                deduped2.append({
                    "start":    raw_segments[i]["start"], "end": end,
                    "text":     " ".join(s["text"] for s in raw_segments[i:j+1]),
                    "duration": round(dur, 1),
                    "headline": raw_segments[i].get("headline", ""),
                })
                break  # pega só o primeiro que encaixa por segmento-âncora
        deduped2 = [c for i, c in enumerate(deduped2)
                    if not any(c["start"] < d["end"] and c["end"] > d["start"]
                               for d in deduped2[:i])]
        if deduped2:
            return deduped2

    # Tentativa 2: dividir por frases
    if transcript:
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        clips, cursor, window, wdur = [], 0.0, [], 0.0
        for sent in sentences:
            dur = len(sent.split()) / 2.5
            window.append(sent); wdur += dur
            if wdur >= target - 10:
                clips.append({
                    "start":    round(cursor, 1),
                    "end":      round(cursor + wdur, 1),
                    "text":     " ".join(window),
                    "duration": round(wdur, 1),
                    "headline": "",
                })
                cursor += wdur; window, wdur = [], 0.0
        if window:  # último pedaço que sobrou
            clips.append({
                "start":    round(cursor, 1),
                "end":      round(cursor + wdur, 1),
                "text":     " ".join(window),
                "duration": round(wdur, 1),
                "headline": "",
            })
        if clips:
            return clips

    # Tentativa 3 (último recurso): um único clipe com tudo
    if raw_segments:
        return [{
            "start":    raw_segments[0]["start"],
            "end":      raw_segments[-1]["end"],
            "text":     " ".join(s["text"] for s in raw_segments),
            "duration": round(raw_segments[-1]["end"] - raw_segments[0]["start"], 1),
            "headline": "",
        }]

    return []


def pick_top_clips(scored, settings):
    n  = settings.get("num_clips", 5)
    ms = settings.get("min_score", 30)
    if not scored:
        return []
    filtered = [c for c in scored if c["score"] >= ms]
    # Se nenhum passou o threshold, ignora o filtro e pega os melhores mesmo assim
    result = (filtered or scored)[:n]
    return result


# ═══════════════════════════════════════════════════════════════
# CORTE DOS CLIPES — com legendas SRT + remoção inteligente de silêncio
# ═══════════════════════════════════════════════════════════════

def cut_clips(video_path, clips, job_id, settings, words_data):
    out_dir       = OUTPUTS/job_id; out_dir.mkdir(exist_ok=True)
    aspect        = settings.get("aspect","9:16")
    crf           = str(settings.get("crf","20"))
    use_captions  = settings.get("captions", True)
    use_silence   = settings.get("remove_silence", True)
    results       = []

    # Detectar resolução original
    probe = subprocess.run(["ffprobe","-v","quiet","-print_format","json",
                            "-show_streams",video_path], capture_output=True, text=True)
    try:
        vs     = next((s for s in json.loads(probe.stdout).get("streams",[])
                       if s.get("codec_type")=="video"), {})
        orig_w = int(vs.get("width",1920))
        orig_h = int(vs.get("height",1080))
    except:
        orig_w, orig_h = 1920, 1080

    for i, clip in enumerate(clips):
        fname      = f"clip_{i+1:02d}_score{clip['score']}.mp4"
        out_path   = out_dir/fname
        raw_path   = out_dir/f"raw_{i+1:02d}.mp4"
        nosil_path = out_dir/f"nosil_{i+1:02d}.mp4"
        srt_path   = out_dir/f"clip_{i+1:02d}.srt"
        cs, ce     = clip["start"], clip["end"]
        duration   = ce - cs

        # PASSO 1 — Recortar segmento bruto
        vf = []
        if aspect == "9:16":
            cw = min(orig_w, int(orig_h*9/16))
            vf += [f"crop={cw}:{orig_h}:(iw-{cw})/2:0", "scale=1080:1920:flags=lanczos"]
        elif aspect == "1:1":
            sq = min(orig_w, orig_h)
            vf += [f"crop={sq}:{sq}:(iw-{sq})/2:(ih-{sq})/2","scale=1080:1080:flags=lanczos"]
        else:
            vf += ["scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease",
                   "pad=1920:1080:-1:-1:color=black"]

        cmd = (["ffmpeg","-y","-ss",str(cs),"-i",video_path,"-t",str(duration),
                "-c:v","libx264","-preset","fast","-crf",crf,
                "-profile:v","high","-level","4.1","-pix_fmt","yuv420p",
                "-c:a","aac","-b:a","192k","-ar","44100"]
               + (["-vf",",".join(vf)] if vf else [])
               + ["-movflags","+faststart", str(raw_path)])
        subprocess.run(cmd, capture_output=True, timeout=600)

        if not raw_path.exists() or raw_path.stat().st_size < 1000:
            continue

        # PASSO 2 — Remover silêncios inteligentemente
        working = smart_silence_removal(raw_path, nosil_path, job_id, i) if use_silence else str(raw_path)

        # PASSO 3 — Gerar arquivo SRT
        has_srt = False
        if use_captions and words_data:
            has_srt = generate_srt(words_data, cs, ce, srt_path)

        # PASSO 4 — Queimar legendas no vídeo
        if has_srt and srt_path.exists():
            srt_esc = str(srt_path).replace("'","\\'")
            if aspect == "9:16":
                style = (f"subtitles={srt_esc}:force_style="
                         "'FontName=Arial,FontSize=22,Bold=1,"
                         "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                         "BackColour=&H80000000,Outline=2,Shadow=1,"
                         "MarginV=60,Alignment=2'")
            else:
                style = (f"subtitles={srt_esc}:force_style="
                         "'FontName=Arial,FontSize=18,Bold=1,"
                         "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                         "BackColour=&H80000000,Outline=2,Shadow=1,"
                         "MarginV=40,Alignment=2'")
            r = subprocess.run([
                "ffmpeg","-y","-i",working,"-vf",style,
                "-c:v","libx264","-preset","slow","-crf",crf,
                "-profile:v","high","-level","4.1","-pix_fmt","yuv420p",
                "-c:a","copy","-movflags","+faststart", str(out_path)
            ], capture_output=True, timeout=600)
            if not out_path.exists() or out_path.stat().st_size < 1000:
                subprocess.run(["cp", working, str(out_path)])  # fallback
        else:
            subprocess.run(["cp", working, str(out_path)])

        # Limpar temporários
        for tmp in [raw_path, nosil_path, srt_path]:
            try: tmp.unlink(missing_ok=True)
            except: pass

        size_mb = round(out_path.stat().st_size/1024/1024, 1) if out_path.exists() else 0
        results.append({
            "filename": fname, "url": f"/clip/{job_id}/{fname}",
            "start": cs, "end": ce, "duration": round(duration,1),
            "score": clip["score"], "grade": clip["grade"],
            "hook": clip["hook"], "flow": clip["flow"],
            "value": clip["value"], "trend": clip["trend"],
            "preview_text": clip["text"][:140]+"...",
            "size_mb": size_mb,
            "has_subtitles": has_srt,
            "silence_removed": use_silence,
        })
    return results


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index(): return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    url  = data.get("url","").strip()
    if not url: return jsonify({"error":"URL obrigatória"}), 400
    if not AAI_KEY: return jsonify({"error":"❌ Chave AssemblyAI não configurada."}), 400
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
        if r.status_code==200:   return jsonify({"ok":True,"msg":"Chave válida!"})
        elif r.status_code==401: return jsonify({"ok":False,"erro":"Chave INVÁLIDA — 401"})
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
