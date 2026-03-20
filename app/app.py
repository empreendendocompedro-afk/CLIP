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
    """
    Grava YOUTUBE_COOKIES em arquivo .txt para o yt-dlp usar.
    O Railway às vezes codifica quebras de linha como \n literal — corrigimos aqui.
    """
    cookie_env = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not cookie_env:
        return False
    # Corrigir \n literal → newline real (problema comum no Railway)
    cookie_env = cookie_env.replace("\\n", "\n").replace("\\r", "")
    # Garantir que começa com o header correto do Netscape
    if not cookie_env.startswith("# Netscape"):
        cookie_env = "# Netscape HTTP Cookie File\n" + cookie_env
    with open(COOKIES_FILE, "w", encoding="utf-8") as f:
        f.write(cookie_env)
    return True

# Configurar cookies ao iniciar E reconfigurar a cada chamada
HAS_COOKIES = setup_cookies()


AAI_KEY    = os.environ.get("ASSEMBLYAI_API_KEY", "")
YTDLP_PROXY = os.environ.get("YTDLP_PROXY", "")  # ex: http://user:pass@ip:porta
MAX_DURATION_SEC = 10800  # 3 horas

# ─── Controle de cancelamento ────────────────────────────────
CANCEL_FLAGS = {}  # job_id → True se cancelado

def is_cancelled(jid):
    return CANCEL_FLAGS.get(jid, False)

def cancel_job(jid):
    CANCEL_FLAGS[jid] = True
    job_update(jid, status="cancelled", message="⛔ Cancelado pelo usuário.")

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
                message=f"❌ Vídeo muito longo ({mins} min). O limite é 3 horas.")
        mins_label = f" ({int(duration_sec//60)} min)" if duration_sec else ""
        job_update(job_id, progress=5, message=f"⬇️ Baixando vídeo{mins_label} (pode levar alguns minutos)...", stage="download")

        # 2. Download
        video_path, dl_error = download_video(url, job_id, settings)
        if not video_path:
            return job_update(job_id, status="error", message=f"❌ Download falhou: {dl_error}")

        # 3. Remuxar vídeo para garantir arquivo limpo e seekável
        if is_cancelled(job_id): return
        job_update(job_id, progress=15, message="🔧 Preparando vídeo...", stage="download")
        video_path = remux_video(video_path, job_id)

        # 4. Áudio
        job_update(job_id, progress=18, message="🔊 Extraindo áudio...", stage="download")
        audio_path = extract_audio(video_path, job_id)
        af = Path(audio_path)
        if not af.exists() or af.stat().st_size < 1000:
            return job_update(job_id, status="error", message="❌ Áudio inválido ou muito curto.")

        # 4. Transcrição
        if is_cancelled(job_id): return
        job_update(job_id, progress=30,
            message="🎙️ Transcrevendo (AssemblyAI)...", stage="transcribe",
            audio_size_kb=round(af.stat().st_size/1024, 1))
        transcript, raw_segments, words_data, aai_err = transcribe_assemblyai(audio_path, job_id)
        if not transcript:
            return job_update(job_id, status="error", message=f"❌ Transcrição falhou: {aai_err}")

        # 5. Extrair energia de áudio
        if is_cancelled(job_id): return
        job_update(job_id, progress=50, message="🎵 Analisando energia e emoção do áudio...", stage="analyze")
        energy_data = extract_audio_energy(audio_path, job_id)

        # 6. Análise inteligente com Claude
        scored = None
        if ANTHROPIC_KEY:
            job_update(job_id, progress=58, message="🤖 Claude analisando blocos semânticos...", stage="analyze")
            scored, claude_err = analyze_with_claude(transcript, words_data, energy_data, settings, job_id)
            if not scored:
                job_update(job_id, claude_fallback_reason=claude_err)

        # Fallback: análise legada com regex + scoring
        if not scored:
            job_update(job_id, progress=62, message="🧠 Analisando semântica (modo clássico)...", stage="analyze")
            segments = analyze_transcript(transcript, raw_segments, settings)
            if not segments:
                return job_update(job_id, status="error",
                    message="❌ Nenhum segmento identificado no vídeo. O vídeo tem fala/diálogo?")
            # Enriquecer com energia de áudio mesmo no fallback
            for seg in segments:
                e_avg, e_peak = energy_for_segment(energy_data, seg["start"], seg["end"])
                seg["energy_avg"]      = e_avg
                seg["energy_peak_val"] = e_peak
            job_update(job_id, progress=68, message="📈 Calculando Virality Score™...", stage="score")
            scored = score_clips_opusclip(segments, transcript)

        job_update(job_id, scored_count=len(scored),
                   top_score=scored[0]["score"] if scored else 0,
                   segments_count=len(scored))

        # 7. Selecionar
        job_update(job_id, progress=78, message="✂️ Selecionando melhores clipes...", stage="score")
        top_clips = pick_top_clips(scored, settings)

        # 8. Render com legendas + sem silêncios
        if is_cancelled(job_id): return
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
    # Recarregar cookies frescos a cada download
    setup_cookies()

    out  = UPLOADS / job_id
    out.mkdir(exist_ok=True)
    tmpl = str(out / "video.%(ext)s")
    crf  = settings.get("crf", "20")

    # Qualidade de download sempre a melhor disponível
    # (a compressão de saída é controlada pelo CRF no render)

    # Base de argumentos comuns
    base = ["yt-dlp", "--no-playlist", "--no-check-certificates",
            "--merge-output-format", "mp4", "-o", tmpl,
            ]

    # Cookies do YouTube (essencial para vídeos com restrição)
    # Proxy removido — YouTube detecta e bloqueia proxies residenciais

    # Adiciona cookies se disponível (cookies de conta BR reforçam o bypass)
    if COOKIES_FILE.exists():
        base += ["--cookies", str(COOKIES_FILE)]

    # Flags Deno para resolver n-challenge
    ejs_flags = ["--remote-components", "ejs:github", "--js-runtimes", "deno"]

    # Formato de melhor qualidade — vídeo e áudio separados mergeados em MP4
    # Ordem de preferência: 1080p VP9/H264 + opus/aac, fallback para 720p, fallback para best
    best_fmt = (
        "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/"  # H264+AAC — máxima qualidade
        "bestvideo[vcodec^=vp9]+bestaudio/"                  # VP9+opus — alta qualidade
        "bestvideo+bestaudio/"                               # qualquer codec — melhor disponível
        "best"                                               # fallback
    )

    strategies = [
        # 1) Web + Deno + melhor qualidade (funciona com cookies)
        base + ejs_flags + ["--extractor-args", "youtube:player_client=web",
                            "-f", best_fmt, url],
        # 2) TV embedded + melhor qualidade disponível
        base + ejs_flags + ["--extractor-args", "youtube:player_client=tv_embedded",
                            "-f", best_fmt, url],
        # 3) Qualquer cliente + melhor qualidade
        base + ejs_flags + ["-f", best_fmt, url],
        # 4) Último recurso sem seleção de formato
        base + ["-f", "best", url],
    ]

    last_err = ""
    for i, cmd in enumerate(strategies):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        job_update(job_id, **{f"dl_log_{i}": (r.stdout+r.stderr)[-2000:]})
        videos = list(out.glob("video.*"))
        if videos: return str(videos[0]), None
        last_err = r.stderr[-600:] if r.stderr else r.stdout[-600:]
    return None, last_err


# ═══════════════════════════════════════════════════════════════
# REMUX — garante MP4 limpo e seekável antes de qualquer operação
# ═══════════════════════════════════════════════════════════════

def remux_video(video_path, job_id):
    """
    Re-mux o vídeo baixado para garantir que está íntegro, seekável
    e com moov atom no início (faststart). Essencial quando o download
    usa clientes alternativos (tv_embedded, mweb) que podem gerar MP4
    fragmentado ou com moov atom no final.
    """
    out_path = Path(video_path).parent / "video_clean.mp4"
    r = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-c", "copy",                  # copia streams sem reencoder
        "-movflags", "+faststart",     # move moov atom para o início
        str(out_path)
    ], capture_output=True, text=True, timeout=1200)

    log = (r.stdout + r.stderr)[-300:]
    job_update(job_id, remux_log=log)

    if out_path.exists() and out_path.stat().st_size > 100000:
        # Detectar codec do arquivo remuxado
        probe2 = subprocess.run([
            "ffprobe","-v","quiet","-print_format","json","-show_streams", str(out_path)
        ], capture_output=True, text=True)
        try:
            streams = json.loads(probe2.stdout).get("streams", [])
            vs = next((s for s in streams if s.get("codec_type")=="video"), {})
            codec_info = f"codec={vs.get('codec_name','?')} {vs.get('width','?')}x{vs.get('height','?')} dur={vs.get('duration','?')}"
        except:
            codec_info = "probe falhou"
        job_update(job_id, remux_codec=codec_info)
        try: Path(video_path).unlink()
        except: pass
        return str(out_path)
    else:
        job_update(job_id, remux_log=log + " | REMUX FALHOU, usando original")
        return video_path


# ═══════════════════════════════════════════════════════════════
# ÁUDIO
# ═══════════════════════════════════════════════════════════════

def extract_audio(video_path, job_id):
    audio_path = UPLOADS / job_id / "audio.mp3"
    r = subprocess.run([
        "ffmpeg","-y","-i",video_path,
        "-ar","16000","-ac","1","-vn","-b:a","32k",
        "-t","10800", str(audio_path)
    ], capture_output=True, timeout=1800)
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
                               headers=headers, data=f, timeout=600)
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

    for attempt in range(720):  # 60 min timeout
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
    return None, [], [], "Timeout (60 min) — AssemblyAI demorou muito. Tente um vídeo menor."


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
    """
    Remove silêncios usando concat demuxer — simples, robusto, sem limite de segmentos.
    Estratégia: detectar silêncios → cortar segmentos de fala → concatenar com -c copy.
    """
    try:
        # 1. Detectar silêncios
        probe = subprocess.run([
            "ffmpeg", "-i", input_path,
            "-af", "silencedetect=noise=-38dB:d=0.5",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=120)

        log            = probe.stderr
        silence_starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
        silence_ends   = [float(m) for m in re.findall(r"silence_end: ([\d.]+)",   log)]

        if not silence_starts:
            return input_path  # sem silêncio detectado

        # 2. Duração total do clipe
        dur_r = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", input_path
        ], capture_output=True, text=True, timeout=30)
        total_dur = float(json.loads(dur_r.stdout).get("format", {}).get("duration", 0))
        if total_dur < 1:
            return input_path

        # 3. Montar intervalos de fala (entre silêncios)
        MIN_SEG = 0.3   # segmento mínimo de fala (evita micro-cortes)
        speech  = []
        cursor  = 0.0

        for ss, se in zip(silence_starts, silence_ends):
            if ss - cursor > MIN_SEG:
                speech.append((round(cursor, 3), round(ss, 3)))
            cursor = se

        if total_dur - cursor > MIN_SEG:
            speech.append((round(cursor, 3), round(total_dur, 3)))

        # Sem segmentos suficientes — não vale cortar
        if len(speech) <= 1:
            return input_path

        # Verificar se há silêncio suficiente para valer o processamento
        total_speech = sum(e - s for s, e in speech)
        if total_speech > total_dur * 0.92:  # menos de 8% de silêncio → não processa
            return input_path

        # 4. Cortar cada segmento de fala
        tmp_dir   = Path(input_path).parent / f"sil_{clip_index}"
        tmp_dir.mkdir(exist_ok=True)
        seg_files = []

        for si, (s, e) in enumerate(speech):
            dur = e - s
            if dur < MIN_SEG:
                continue
            seg_out = tmp_dir / f"s{si:04d}.mp4"
            r = subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(s), "-to", str(e),
                "-i", input_path,
                "-c", "copy",           # cópia sem reencoder — muito mais rápido
                "-avoid_negative_ts", "make_zero",
                str(seg_out)
            ], capture_output=True, timeout=60)
            if seg_out.exists() and seg_out.stat().st_size > 200:
                seg_files.append(str(seg_out))

        if len(seg_files) < 2:
            try: __import__("shutil").rmtree(tmp_dir)
            except: pass
            return input_path

        # 5. Concatenar com concat demuxer (-c copy, sem reencoder, sem xfade)
        concat_txt = tmp_dir / "list.txt"
        with open(concat_txt, "w") as f:
            for sf in seg_files:
                f.write("file '" + sf + "'\n")

        r = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ], capture_output=True, text=True, timeout=300)

        # Limpar temporários
        try: __import__("shutil").rmtree(tmp_dir)
        except: pass

        if Path(output_path).exists() and Path(output_path).stat().st_size > 5000:
            return str(output_path)
        return input_path

    except Exception as ex:
        try: __import__("shutil").rmtree(Path(input_path).parent / f"sil_{clip_index}")
        except: pass
        return input_path


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

# ═══════════════════════════════════════════════════════════════
# EXTRAÇÃO DE ENERGIA DE ÁUDIO
# ═══════════════════════════════════════════════════════════════

def extract_audio_energy(audio_path, job_id):
    """
    Usa FFmpeg astats para extrair energia RMS por segundo do áudio.
    Retorna lista de {second, rms_db, rms_norm} onde rms_norm é 0-1.
    """
    try:
        r = subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-"
        ], capture_output=True, text=True, timeout=120)

        energy = []
        second = 0
        for line in r.stderr.split("\n"):
            if "lavfi.astats.Overall.RMS_level" in line:
                try:
                    val = float(line.split("=")[-1].strip())
                    if val > -100:  # ignora silêncio total
                        energy.append({"second": second, "rms_db": val})
                    second += 1
                except:
                    pass

        if not energy:
            return []

        # Normalizar para 0-1
        min_db = min(e["rms_db"] for e in energy)
        max_db = max(e["rms_db"] for e in energy)
        rng    = max(max_db - min_db, 1)
        for e in energy:
            e["rms_norm"] = round((e["rms_db"] - min_db) / rng, 3)

        job_update(job_id, audio_energy_points=len(energy))
        return energy

    except Exception as ex:
        job_update(job_id, audio_energy_error=str(ex))
        return []


def energy_for_segment(energy_data, start_sec, end_sec):
    """Retorna energia média e pico de um intervalo de tempo."""
    pts = [e["rms_norm"] for e in energy_data
           if start_sec <= e["second"] <= end_sec]
    if not pts:
        return 0.5, 0.5
    return round(sum(pts)/len(pts), 3), round(max(pts), 3)


# ═══════════════════════════════════════════════════════════════
# ANÁLISE INTELIGENTE COM CLAUDE
# ═══════════════════════════════════════════════════════════════

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def analyze_with_claude(transcript, words_data, energy_data, settings, job_id):
    """
    Envia transcrição + dados de energia para Claude claude-sonnet-4-5.
    Claude identifica blocos semânticos completos: raciocínios, pares
    pergunta/resposta, picos emocionais, viradas de conversa.
    Retorna lista de segmentos com start/end/score/reasoning.
    """
    if not ANTHROPIC_KEY:
        return None, "ANTHROPIC_API_KEY não configurada"

    target   = settings.get("clip_duration", 60)
    num      = settings.get("num_clips", 5)
    min_dur  = max(15, target - 25)
    max_dur  = target + 30

    # Construir mapa de energia por segundo (resumido)
    energy_summary = ""
    if energy_data:
        # Resumir energia em blocos de 5 segundos para não sobrecarregar o prompt
        peaks = []
        for i in range(0, len(energy_data), 5):
            blk = energy_data[i:i+5]
            avg = sum(b["rms_norm"] for b in blk) / len(blk)
            t   = blk[0]["second"]
            if avg > 0.65:  # só picos relevantes
                peaks.append(f"{t}s({avg:.2f})")
        if peaks:
            energy_summary = "\nPICOS DE ENERGIA (segundos com alta intensidade vocal): " + ", ".join(peaks[:30])

    # Transcrição com timestamps — uma linha por palavra para Claude entender timing
    # Agrupar palavras em frases de ~10 palavras para não estourar o contexto
    timed_lines = []
    if words_data:
        buf, buf_start = [], None
        for w in words_data:
            if buf_start is None:
                buf_start = w.get("start", 0) / 1000
            buf.append(w.get("text", ""))
            if len(buf) >= 10:
                buf_end = w.get("end", 0) / 1000
                timed_lines.append(f"[{buf_start:.1f}s-{buf_end:.1f}s] {' '.join(buf)}")
                buf, buf_start = [], None
        if buf and buf_start is not None:
            last_end = words_data[-1].get("end", 0) / 1000
            timed_lines.append(f"[{buf_start:.1f}s-{last_end:.1f}s] {' '.join(buf)}")

    timed_transcript = "\n".join(timed_lines[:800])  # máximo 800 linhas (~3h de vídeo)

    prompt = f"""Você é um especialista em criação de conteúdo viral para redes sociais.
Analise a transcrição abaixo e identifique exatamente {num} clipes virais.

REGRAS CRÍTICAS:
1. NUNCA corte no meio de um raciocínio, frase ou argumento
2. Cada clipe deve ter início e fim SEMÂNTICO — começa quando uma ideia começa, termina quando ela termina
3. Duração alvo: {target}s (mínimo {min_dur}s, máximo {max_dur}s)
4. Priorize em ordem: picos de energia emocional do locutor > viradas/surpresas > pares pergunta+resposta completos > raciocínios completos
5. Clipes não podem se sobrepor
{energy_summary}

TRANSCRIÇÃO COM TIMESTAMPS:
{timed_transcript}

Responda APENAS com JSON válido, sem markdown, sem explicação:
{{
  "clips": [
    {{
      "start": 12.5,
      "end": 74.2,
      "hook": "primeira frase que prende atenção",
      "score": 85,
      "reasoning": "por que este clipe é viral",
      "energy_peak": true,
      "has_question_answer": false,
      "has_turning_point": true
    }}
  ]
}}"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-5",
                "max_tokens": 2000,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60
        )
        job_update(job_id, claude_status=resp.status_code)

        if resp.status_code != 200:
            return None, f"Claude API HTTP {resp.status_code}: {resp.text[:200]}"

        raw = resp.json()["content"][0]["text"].strip()
        # Remove markdown se vier
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        clips = data.get("clips", [])
        if not clips:
            return None, "Claude não retornou clipes"

        # Converter para formato padrão do pipeline
        segments = []
        for c in clips:
            start = float(c.get("start", 0))
            end   = float(c.get("end",   0))
            dur   = end - start
            if dur < 10:
                continue
            # Buscar texto do segmento nos words_data
            seg_words = [w["text"] for w in words_data
                         if start <= w.get("start",0)/1000 <= end + 0.5] if words_data else []
            text = " ".join(seg_words) or c.get("hook", "")

            # Calcular energia média do segmento
            e_avg, e_peak = energy_for_segment(energy_data, start, end)

            segments.append({
                "start":             start,
                "end":               end,
                "text":              text,
                "duration":          round(dur, 1),
                "headline":          c.get("hook", ""),
                "score":             min(99, int(c.get("score", 50))),
                "grade":             score_to_grade(int(c.get("score", 50))),
                "hook":              min(25, int(c.get("score", 50) * 0.28)),
                "flow":              min(25, int(c.get("score", 50) * 0.26)),
                "value":             min(25, int(c.get("score", 50) * 0.25)),
                "trend":             min(25, int(c.get("score", 50) * 0.21)),
                "reasoning":         c.get("reasoning", ""),
                "energy_avg":        e_avg,
                "energy_peak_val":   e_peak,
                "has_energy_peak":   bool(c.get("energy_peak", False)),
                "has_qa":            bool(c.get("has_question_answer", False)),
                "has_turning_point": bool(c.get("has_turning_point", False)),
            })

        job_update(job_id, claude_clips_found=len(segments))
        return sorted(segments, key=lambda x: x["score"], reverse=True), None

    except json.JSONDecodeError as e:
        return None, f"Claude retornou JSON inválido: {e} | raw: {raw[:200]}"
    except Exception as e:
        return None, f"Erro Claude: {type(e).__name__}: {e}"


def score_to_grade(score):
    if score >= 75: return "🔥 Viral"
    if score >= 55: return "⚡ Alto"
    if score >= 35: return "👍 Médio"
    return "📉 Baixo"


# ═══════════════════════════════════════════════════════════════
# SCORING LEGADO (fallback se Claude não disponível)
# ═══════════════════════════════════════════════════════════════

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ═══════════════════════════════════════════════════════════════
# SCORING LEGADO (fallback quando Claude não disponível)
# ═══════════════════════════════════════════════════════════════
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
    """
    Constroi segmentos de duracao adequada.
    Detecta se capitulos sao curtos demais e usa janela deslizante.
    """
    target  = settings.get("clip_duration", 60)
    min_dur = max(20, target - 30)
    max_dur = target + 40

    # Verificar se capitulos da AssemblyAI sao uteis (media >= 20s)
    use_chapters = False
    if raw_segments and len(raw_segments) >= 2:
        total_span = raw_segments[-1]["end"] - raw_segments[0]["start"]
        avg_dur    = total_span / len(raw_segments)
        use_chapters = avg_dur >= 20

    # Metodo 1: combinar capitulos longos o suficiente
    if use_chapters:
        candidates, deduped = [], []
        for i in range(len(raw_segments)):
            for j in range(i + 1, len(raw_segments)):
                dur = raw_segments[j]["end"] - raw_segments[i]["start"]
                if dur < min_dur: continue
                if dur > max_dur: break
                candidates.append({
                    "start":    raw_segments[i]["start"],
                    "end":      raw_segments[j]["end"],
                    "text":     " ".join(s["text"] for s in raw_segments[i:j+1]),
                    "duration": round(dur, 1),
                    "headline": raw_segments[i].get("headline", ""),
                })
        for c in sorted(candidates, key=lambda x: x["duration"]):
            if not any(c["start"] < d["end"] and c["end"] > d["start"] for d in deduped):
                deduped.append(c)
        if deduped:
            return deduped

    # Metodo 2: janela deslizante usando velocidade de fala real
    if raw_segments:
        total_dur   = raw_segments[-1]["end"] - raw_segments[0]["start"]
        total_words = sum(len(s["text"].split()) for s in raw_segments)
        wps         = max(1.0, total_words / max(1, total_dur))

        clips     = []
        buf_words = []
        buf_start = raw_segments[0]["start"]

        for seg in raw_segments:
            for word in seg["text"].split():
                buf_words.append(word)
                elapsed = len(buf_words) / wps
                if elapsed >= target:
                    end_t = buf_start + elapsed
                    clips.append({
                        "start":    round(buf_start, 1),
                        "end":      round(end_t, 1),
                        "text":     " ".join(buf_words),
                        "duration": round(elapsed, 1),
                        "headline": buf_words[0] if buf_words else "",
                    })
                    buf_start = end_t
                    buf_words = []

        if buf_words:
            elapsed = len(buf_words) / wps
            if elapsed >= min_dur:
                clips.append({
                    "start":    round(buf_start, 1),
                    "end":      round(buf_start + elapsed, 1),
                    "text":     " ".join(buf_words),
                    "duration": round(elapsed, 1),
                    "headline": "",
                })

        if clips:
            return clips

    # Metodo 3: texto bruto
    if transcript:
        words  = transcript.split()
        wps    = 2.5
        clips  = []
        cursor = 0.0
        step   = int(target * wps)
        for i in range(0, len(words), step):
            chunk = words[i:i+step]
            if not chunk: break
            dur = len(chunk) / wps
            if dur >= min_dur:
                clips.append({
                    "start": round(cursor, 1), "end": round(cursor+dur, 1),
                    "text": " ".join(chunk), "duration": round(dur, 1), "headline": "",
                })
            cursor += dur
        if clips:
            return clips

    # Fallback: tudo em um clipe
    if raw_segments:
        return [{"start": raw_segments[0]["start"], "end": raw_segments[-1]["end"],
                 "text": " ".join(s["text"] for s in raw_segments),
                 "duration": round(raw_segments[-1]["end"]-raw_segments[0]["start"],1),
                 "headline": ""}]
    return []


def pick_top_clips(scored, settings):
    n          = settings.get("num_clips", 5)
    ms         = settings.get("min_score", 30)
    target_dur = settings.get("clip_duration", 60)
    min_dur    = max(15, target_dur - 30)  # mínimo absoluto

    if not scored:
        return []

    # Descartar segmentos com duração absurda (< 5s ou > 20min)
    valid = [c for c in scored
             if (c["end"] - c["start"]) >= 5
             and (c["end"] - c["start"]) <= 1200]

    # Se todos foram descartados, usar originais (algo errado nos timestamps)
    if not valid:
        valid = scored

    filtered = [c for c in valid if c["score"] >= ms]
    result   = (filtered or valid)[:n]
    return result


# ═══════════════════════════════════════════════════════════════
# CORTE DOS CLIPES — com legendas SRT + remoção inteligente de silêncio
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# FACE DETECTION & TRACKING
# ═══════════════════════════════════════════════════════════════

def detect_and_track_faces(video_path, clip_start, clip_end, orig_w, orig_h,
                           aspect, crf, out_path, job_id, clip_idx):
    """
    Face tracking com MediaPipe (Google) — muito mais preciso que Haar Cascade.
    Detecta rostos reais com confiança mínima, ignora objetos/equipamentos.
    Com múltiplos rostos: segue quem tem mais movimento labial.
    """
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
    except ImportError as e:
        job_update(job_id, **{f"face_{clip_idx}_status": f"import error: {e}"})
        return False

    # ── Dimensões de crop ───────────────────────────────────────
    ar     = {"9:16": 9/16, "1:1": 1.0, "16:9": 16/9}.get(aspect, 9/16)
    crop_w = int(orig_h * ar)
    crop_h = orig_h
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(orig_w / ar)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    half_w  = crop_w // 2

    if aspect == "9:16":
        out_w, out_h = 1080, 1920
    elif aspect == "1:1":
        out_w = out_h = 1080
    else:
        out_w, out_h = 1920, 1080

    # ── Inicializar MediaPipe Face Detection ────────────────────
    # Suporta API legada (< 0.10) e nova (>= 0.10)
    try:
        # API legada
        mp_face    = mp.solutions.face_detection
        face_model = mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.55)
        use_legacy = True
    except AttributeError:
        # Nova API (mediapipe >= 0.10)
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, tempfile, os
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            model_path = "/tmp/face_detector.tflite"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(model_url, model_path)
            base_opts  = mp_tasks.BaseOptions(model_asset_path=model_path)
            det_opts   = mp_vision.FaceDetectorOptions(
                base_options=base_opts, min_detection_confidence=0.55)
            face_model = mp_vision.FaceDetector.create_from_options(det_opts)
            use_legacy = False
        except Exception as e2:
            job_update(job_id, **{f"face_{clip_idx}_status": f"mediapipe init error: {e2}"})
            return False

    cap      = cv2.VideoCapture(str(video_path))
    duration = clip_end - clip_start
    samples  = []   # (t_offset, cx)
    prev_mouths = {}
    interval = 0.5
    t = clip_start

    while t <= clip_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        offset = round(t - clip_start, 2)

        # Detectar rostos com a API correta
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = []
        if use_legacy:
            results = face_model.process(rgb)
            if results.detections:
                for det in results.detections:
                    bb   = det.location_data.relative_bounding_box
                    detections.append({
                        "xmin": bb.xmin, "ymin": bb.ymin,
                        "w": bb.width,   "h": bb.height,
                        "conf": det.score[0]
                    })
        else:
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_model.detect(mp_img)
            if results.detections:
                for det in results.detections:
                    bb = det.bounding_box
                    detections.append({
                        "xmin": bb.origin_x / orig_w,
                        "ymin": bb.origin_y / orig_h,
                        "w":    bb.width    / orig_w,
                        "h":    bb.height   / orig_h,
                        "conf": det.categories[0].score if det.categories else 0.6
                    })

        if not detections:
            samples.append((offset, None))
            t += interval
            continue

        faces_px = []
        for d in detections:
            fx   = int(d["xmin"] * orig_w)
            fy   = int(d["ymin"] * orig_h)
            fw   = int(d["w"]    * orig_w)
            fh   = int(d["h"]    * orig_h)
            conf = d["conf"]
            if fw < 20 or fh < 20:
                continue
            if fy + fh * 0.5 > orig_h * 0.90:
                continue
            # Ignorar inserções/thumbnails no canto inferior direito
            rel_cx = (fx + fw/2) / max(orig_w, 1)
            rel_cy = (fy + fh/2) / max(orig_h, 1)
            if rel_cx > 0.68 and rel_cy > 0.68:
                continue
            cx = max(0, min(orig_w, fx + fw // 2))
            faces_px.append({"cx": cx, "fx": fx, "fy": fy, "fw": fw, "fh": fh, "conf": conf})

        if not faces_px:
            samples.append((offset, None))
            t += interval
            continue

        if len(faces_px) == 1:
            samples.append((offset, faces_px[0]["cx"]))
        else:
            # Múltiplos rostos — escolhe pelo movimento labial
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            best_cx    = None
            best_score = -1

            for face in faces_px:
                fx, fy, fw, fh = face["fx"], face["fy"], face["fw"], face["fh"]
                cx_face = face["cx"]

                # Região da boca: 55-100% do rosto
                my1 = max(0, fy + int(fh * 0.55))
                my2 = min(orig_h, fy + fh)
                mx1 = max(0, fx)
                mx2 = min(orig_w, fx + fw)
                mouth = gray[my1:my2, mx1:mx2]
                if mouth.size == 0:
                    continue

                key = cx_face // 40
                lip_score = 0
                if key in prev_mouths and prev_mouths[key].shape == mouth.shape:
                    lip_score = int(np.mean(cv2.absdiff(mouth, prev_mouths[key])))
                prev_mouths[key] = mouth.copy()

                # Penalizar rostos pequenos (inserções) — locutor real é maior
                size_penalty = 0 if (fw * fh) > (orig_w * orig_h * 0.015) else -200
                score = lip_score * 8 + int(face["conf"] * 100) + fw * fh // 200 + size_penalty
                if score > best_score:
                    best_score = score
                    best_cx    = cx_face

            samples.append((offset, best_cx or faces_px[0]["cx"]))

        t += interval

    cap.release()
    try: face_model.close()
    except: pass

    # ── Construir timeline ──────────────────────────────────────
    face_samples = [(t, cx) for t, cx in samples if cx is not None]
    pct          = len(face_samples) / max(1, len(samples))

    job_update(job_id, **{
        f"face_{clip_idx}_frames":   len(samples),
        f"face_{clip_idx}_detected": len(face_samples),
        f"face_{clip_idx}_pct":      f"{int(pct*100)}%",
    })

    if pct < 0.05 or not face_samples:
        job_update(job_id, **{f"face_{clip_idx}_status": f"sem rostos ({int(pct*100)}%)"})
        return False

    # Suavizar com mediana
    cx_raw = [cx for _, cx in face_samples]
    smoothed_cx = []
    W = 5
    for i in range(len(cx_raw)):
        w = cx_raw[max(0, i-W):i+W+1]
        smoothed_cx.append(int(sorted(w)[len(w)//2]))

    # Interpolar frames sem rosto
    last_good = smoothed_cx[0]
    filled_cx = {}
    si = 0
    for t_off, cx_v in samples:
        key = round(t_off, 1)
        if cx_v is not None and si < len(smoothed_cx):
            last_good = smoothed_cx[si]
            si += 1
        filled_cx[key] = last_good

    # Âncoras por cluster
    all_cx  = list(filled_cx.values())
    mid     = orig_w // 2
    left_cx  = [c for c in all_cx if c < mid]
    right_cx = [c for c in all_cx if c >= mid]
    anc_l = int(sum(left_cx)  / len(left_cx))  if left_cx  else mid // 2
    anc_r = int(sum(right_cx) / len(right_cx)) if right_cx else mid + mid // 2

    def snap(cx):
        return anc_l if abs(cx - anc_l) <= abs(cx - anc_r) else anc_r

    MIN_DUR   = 5.0
    raw_segs  = []
    t_keys    = sorted(filled_cx.keys())
    seg_start = 0.0
    cur_anc   = snap(filled_cx[t_keys[0]])

    for tk in t_keys[1:]:
        new_anc = snap(filled_cx[tk])
        if new_anc != cur_anc:
            raw_segs.append({"start": seg_start, "end": tk, "cx": cur_anc})
            seg_start = tk
            cur_anc   = new_anc
    raw_segs.append({"start": seg_start, "end": duration, "cx": cur_anc})

    segments = []
    for seg in raw_segs:
        dur = seg["end"] - seg["start"]
        if segments and segments[-1]["cx"] == seg["cx"]:
            segments[-1]["end"] = seg["end"]
        elif dur < MIN_DUR and segments:
            segments[-1]["end"] = seg["end"]
        else:
            segments.append(dict(seg))
    if segments:
        segments[-1]["end"] = duration

    # ── Render ──────────────────────────────────────────────────
    tmp_dir   = Path(out_path).parent / f"ft_{clip_idx}"
    tmp_dir.mkdir(exist_ok=True)
    seg_files = []

    for si, seg in enumerate(segments):
        seg_dur = seg["end"] - seg["start"]
        if seg_dur < 0.1:
            continue
        cx = seg["cx"]
        x  = max(0, min(orig_w - crop_w, cx - half_w))
        x -= x % 2

        vf = f"crop={crop_w}:{crop_h}:{x}:0,scale={out_w}:{out_h}:flags=lanczos,setsar=1"
        seg_path = tmp_dir / f"s{si:03d}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(clip_start + seg["start"]), "-i", str(video_path),
            "-t", str(seg_dur), "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-threads", "2",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart", str(seg_path)
        ], capture_output=True, text=True, timeout=600)

        if seg_path.exists() and seg_path.stat().st_size > 500:
            seg_files.append(str(seg_path))
            job_update(job_id, **{f"face_{clip_idx}_seg{si}":
                f"cx={cx} x={x} dur={round(seg_dur,1)}s"})

    if not seg_files:
        try: __import__("shutil").rmtree(tmp_dir)
        except: pass
        return False

    out_path = Path(out_path)
    if len(seg_files) == 1:
        __import__("shutil").copy(seg_files[0], str(out_path))
    else:
        concat_txt = tmp_dir / "concat.txt"
        with open(concat_txt, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt), "-c", "copy",
            "-movflags", "+faststart", str(out_path)
        ], capture_output=True, text=True, timeout=300)

    try: __import__("shutil").rmtree(tmp_dir)
    except: pass

    success = out_path.exists() and out_path.stat().st_size > 1000
    job_update(job_id, **{f"face_{clip_idx}_status":
        f"OK — {len(seg_files)} seg(s)" if success else "FALHOU"})
    return success



# ═══════════════════════════════════════════════════════════════
# ESTILOS DE LEGENDA
# ═══════════════════════════════════════════════════════════════

def hex_to_ass_color(hex_color, alpha="00"):
    """Converte #RRGGBB para formato ASS &HAABBGGRR"""
    hex_color = hex_color.lstrip("#")
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{alpha}{b}{g}{r}"

def build_subtitle_filter(srt_path, aspect, style_id, font_size, text_color, bg_opacity):
    """
    Gera o filtro FFmpeg subtitles com o estilo escolhido.
    style_id: 1-15
    font_size: 8-40
    text_color: #RRGGBB
    bg_opacity: 0-100 (transparência do fundo)
    """
    srt_esc    = str(srt_path).replace(":", "\\:")
    margin_v   = "60" if aspect == "9:16" else "40"
    fs         = max(8, min(40, int(font_size)))
    
    # Cor do texto em ASS
    text_ass   = hex_to_ass_color(text_color, "00")
    
    # Opacidade do fundo: 0=opaco, FF=transparente
    bg_alpha   = format(int((1 - bg_opacity/100) * 255), "02X")
    bg_ass     = f"&H{bg_alpha}000000"
    
    white      = "&H00FFFFFF"
    black      = "&H00000000"
    yellow     = "&H0000FFFF"
    shadow_clr = "&H80000000"

    styles = {
        # Estilo 1: Padrão branco com borda preta
        1: (f"FontName=Arial,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"Outline=2,Shadow=1,MarginV={margin_v},Alignment=2"),

        # Estilo 2: Fundo escuro semitransparente
        2: (f"FontName=Arial,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"BackColour={bg_ass},BorderStyle=3,"
            f"Outline=0,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 3: Caixa alta sem borda (estilo TikTok)
        3: (f"FontName=Arial,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"BackColour={bg_ass},BorderStyle=3,"
            f"Outline=1,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 4: Sombra suave (estilo cinema)
        4: (f"FontName=Georgia,FontSize={fs},Bold=0,Italic=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"Outline=1,Shadow=2,MarginV={margin_v},Alignment=2"),

        # Estilo 5: Negrito grande sem fundo (YouTube)
        5: (f"FontName=Impact,FontSize={fs},Bold=0,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"Outline=3,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 6: Fonte monoespaçada (podcast/tech)
        6: (f"FontName=Courier New,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"BackColour={bg_ass},BorderStyle=3,"
            f"Outline=0,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 7: Topo do vídeo (para legendas no topo)
        7: (f"FontName=Arial,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"BackColour={bg_ass},BorderStyle=3,"
            f"Outline=1,Shadow=0,MarginV=20,Alignment=8"),

        # Estilo 8: Minimalista fino
        8: (f"FontName=Arial,FontSize={fs},Bold=0,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"Outline=1,Shadow=1,MarginV={margin_v},Alignment=2"),

        # Estilo 9: Reels/Instagram — fonte grossa centralizada
        9: (f"FontName=Arial Black,FontSize={fs},Bold=1,"
            f"PrimaryColour={text_ass},OutlineColour={black},"
            f"Outline=3,Shadow=2,MarginV={margin_v},Alignment=2"),

        # Estilo 10: Fundo branco texto escuro (contraste invertido)
        10: (f"FontName=Arial,FontSize={fs},Bold=1,"
             f"PrimaryColour={hex_to_ass_color('000000')},"
             f"OutlineColour={hex_to_ass_color('FFFFFF')},"
             f"BackColour=&H20FFFFFF,BorderStyle=3,"
             f"Outline=2,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 11: Neon (brilho de cor)
        11: (f"FontName=Arial,FontSize={fs},Bold=1,"
             f"PrimaryColour={text_ass},OutlineColour={text_ass},"
             f"Outline=4,Shadow=0,MarginV={margin_v},Alignment=2"),

        # Estilo 12: Clássico TV (sem fundo, sombra dupla)
        12: (f"FontName=Times New Roman,FontSize={fs},Bold=0,"
             f"PrimaryColour={text_ass},OutlineColour={black},"
             f"Outline=1,Shadow=3,MarginV={margin_v},Alignment=2"),

        # Estilo 13: Faixa colorida no fundo
        13: (f"FontName=Arial,FontSize={fs},Bold=1,"
             f"PrimaryColour={hex_to_ass_color('FFFFFF')},"
             f"BackColour={hex_to_ass_color(text_color.lstrip('#'), bg_alpha)},"
             f"BorderStyle=3,Outline=0,Shadow=0,"
             f"MarginV={margin_v},Alignment=2"),

        # Estilo 14: Subtítulo editorial (menor, elegante)
        14: (f"FontName=Verdana,FontSize={max(6,fs-4)},Bold=0,Italic=1,"
             f"PrimaryColour={text_ass},OutlineColour={black},"
             f"Outline=1,Shadow=1,MarginV={margin_v},Alignment=2"),

        # Estilo 15: Estilo documentário (uppercase implícito no SRT)
        15: (f"FontName=Helvetica,FontSize={fs},Bold=1,"
             f"PrimaryColour={text_ass},OutlineColour={black},"
             f"BackColour={bg_ass},BorderStyle=3,"
             f"Outline=2,Shadow=2,MarginV={margin_v},Alignment=2"),
    }

    chosen = styles.get(style_id, styles[1])
    return f"subtitles={srt_esc}:force_style='{chosen}'"

def cut_clips(video_path, clips, job_id, settings, words_data):
    out_dir      = OUTPUTS / job_id
    out_dir.mkdir(exist_ok=True)
    aspect       = settings.get("aspect", "9:16")
    crf          = str(settings.get("crf", "20"))
    use_captions   = settings.get("captions", True)
    caption_style  = settings.get("caption_style", 2)
    caption_size   = settings.get("caption_size", 14)
    caption_color  = settings.get("caption_color", "#FFFFFF")
    caption_bg     = settings.get("caption_bg", 60)
    use_silence    = settings.get("remove_silence", True)
    use_face_track = settings.get("face_tracking", True)
    results        = []

    # Detectar resolução original
    try:
        probe  = subprocess.run(["ffprobe","-v","quiet","-print_format","json",
                                 "-show_streams", video_path],
                                capture_output=True, text=True)
        vs     = next((s for s in json.loads(probe.stdout).get("streams",[])
                       if s.get("codec_type") == "video"), {})
        orig_w = int(vs.get("width",  1920))
        orig_h = int(vs.get("height", 1080))
    except:
        orig_w, orig_h = 1920, 1080

    for i, clip in enumerate(clips):
        fname    = f"clip_{i+1:02d}_score{clip['score']}.mp4"
        out_path = out_dir / fname
        tmp_path = out_dir / f"tmp_{i+1:02d}.mp4"
        srt_path = out_dir / f"sub_{i+1:02d}.srt"
        cs, ce   = clip["start"], clip["end"]
        duration = ce - cs

        # ── Filtro de vídeo com crop personalizado ───────────────
        vf = []
        cx = settings.get("crop_x", 0)
        cy = settings.get("crop_y", 0)
        cw = settings.get("crop_w", 1)
        ch = settings.get("crop_h", 1)

        # Verificar se o usuário ajustou o crop manualmente (tolerância de 2%)
        # Se face_tracking ativo e crop neutro (0,0,1,1) → usar face tracking
        user_cropped = not (cx < 0.02 and cy < 0.02 and cw > 0.96 and ch > 0.96)
        job_update(job_id, **{f"crop_debug_{i}":
            f"cx={cx:.2f} cy={cy:.2f} cw={cw:.2f} ch={ch:.2f} user_cropped={user_cropped}"})

        if user_cropped:
            # Usar as coordenadas exatas do crop preview do usuário
            px = int(orig_w * cx)
            py = int(orig_h * cy)
            pw = int(orig_w * cw)
            ph = int(orig_h * ch)
            # Garantir valores pares (requisito do libx264)
            px = px - (px % 2); py = py - (py % 2)
            pw = pw - (pw % 2); ph = ph - (ph % 2)
            pw = max(2, min(pw, orig_w - px))
            ph = max(2, min(ph, orig_h - py))
            vf.append(f"crop={pw}:{ph}:{px}:{py}")
            # Scale para output — não upscalar além de 2x o crop
            max_out_w = min(1080, pw * 2); max_out_w -= max_out_w % 2
            out_h = int(max_out_w * ph / pw); out_h -= out_h % 2
            vf.append(f"scale={max_out_w}:{out_h}:flags=lanczos")
            vf.append("setsar=1")   # pixels quadrados
        else:
            # Crop automático baseado no aspect ratio selecionado
            # Não upscalar além de 2x o source (qualidade sem sentido acima disso)
            if aspect == "9:16":
                auto_cw = min(orig_w, int(orig_h * 9 / 16))
                vf += [f"crop={auto_cw}:{orig_h}:(iw-{auto_cw})/2:0",
                       "scale=1080:1920:flags=lanczos",
                       "setsar=1"]
            elif aspect == "1:1":
                sq = min(orig_w, orig_h)
                vf += [f"crop={sq}:{sq}:(iw-{sq})/2:(ih-{sq})/2",
                       "scale=1080:1080:flags=lanczos",
                       "setsar=1"]
            else:
                vf += ["scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease",
                       "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                       "setsar=1"]

        # PASSO 1: Cortar segmento (com ou sem face tracking)
        face_tracked = False
        if use_face_track and not user_cropped:
            job_update(job_id, message=f"🎯 Face tracking clipe {i+1}...",
                       progress=87 + i)
            face_tracked = detect_and_track_faces(
                video_path, cs, ce, orig_w, orig_h,
                aspect, crf, tmp_path, job_id, i
            )

        if not face_tracked:
            # Fallback: crop estático (manual ou automático)
            cmd1 = (["ffmpeg","-y",
                     "-ss",str(cs),"-i",video_path,"-t",str(duration),
                     "-c:v","libx264",
                     "-preset","veryfast",
                     "-crf","18",
                     "-profile:v","high","-level","4.1",
                     "-pix_fmt","yuv420p",
                     "-threads","2",
                     "-c:a","aac","-b:a","192k","-ar","44100",
                     ]
                    + (["-vf", ",".join(vf)] if vf else [])
                    + ["-movflags","+faststart", str(tmp_path)])
            r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
            full_log = r1.stdout + r1.stderr
            job_update(job_id, **{f"cut_log_{i}": full_log[:600]+"\n---\n"+full_log[-200:],
                                   f"cut_rc_{i}": r1.returncode})

        if not tmp_path.exists() or tmp_path.stat().st_size < 1000:
            job_update(job_id, **{f"clip_{i}_status": "FALHOU no corte"})
            continue

        # PASSO 2: Remover silêncios
        working = str(tmp_path)
        if use_silence:
            nosil = out_dir / f"nosil_{i+1:02d}.mp4"
            working = smart_silence_removal(str(tmp_path), nosil, job_id, i)

        # PASSO 3: Gerar SRT
        has_srt = False
        if use_captions and words_data:
            has_srt = generate_srt(words_data, cs, ce, srt_path)

        # PASSO 4: Queimar legendas (ou copiar sem legenda)
        if has_srt and srt_path.exists():
            style = build_subtitle_filter(
                srt_path, aspect,
                caption_style, caption_size,
                caption_color, caption_bg
            )
            r4 = subprocess.run([
                "ffmpeg","-y","-i",working,"-vf",style,
                "-c:v","libx265","-preset","ultrafast","-crf",crf,
                "-tag:v","hvc1","-pix_fmt","yuv420p",
                "-threads","2","-x265-params","log-level=error",
                "-c:a","copy","-movflags","+faststart", str(out_path)
            ], capture_output=True, text=True, timeout=1800)
            job_update(job_id, **{f"sub_log_{i}": (r4.stdout+r4.stderr)[-400:]})
            if not out_path.exists() or out_path.stat().st_size < 1000:
                subprocess.run(["cp", working, str(out_path)], capture_output=True)
                job_update(job_id, **{f"clip_{i}_status": "sem legenda (sub falhou)"})
            else:
                job_update(job_id, **{f"clip_{i}_status": "OK com legenda"})
        else:
            # Encode final em H265 para qualidade máxima
            r_enc = subprocess.run([
                "ffmpeg", "-y", "-i", working,
                "-c:v", "libx265", "-preset", "ultrafast", "-crf", crf,
                "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
                "-threads", "2", "-x265-params", "log-level=error",
                "-c:a", "copy", "-movflags", "+faststart", str(out_path)
            ], capture_output=True, text=True, timeout=1800)
            if not out_path.exists() or out_path.stat().st_size < 1000:
                subprocess.run(["cp", working, str(out_path)], capture_output=True)
            job_update(job_id, **{f"clip_{i}_status": "OK sem legenda"})

        # Limpar temporários
        for tmp in [tmp_path, srt_path, out_dir/f"nosil_{i+1:02d}.mp4"]:
            try: tmp.unlink(missing_ok=True)
            except: pass

        exists  = out_path.exists() and out_path.stat().st_size > 1000
        size_mb = round(out_path.stat().st_size/1024/1024, 1) if exists else 0
        if not exists:
            job_update(job_id, **{f"clip_{i}_status": "FALHOU — arquivo não criado"})
            continue

        results.append({
            "filename": fname, "url": f"/clip/{job_id}/{fname}",
            "start": cs, "end": ce, "duration": round(duration,1),
            "score": clip["score"], "grade": clip["grade"],
            "hook": clip["hook"], "flow": clip["flow"],
            "value": clip["value"], "trend": clip["trend"],
            "preview_text": clip["text"][:140]+"...",
            "size_mb": size_mb, "has_subtitles": has_srt,
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
        "captions":       bool(data.get("captions", True)),
        "caption_style":  int(data.get("caption_style", 2)),
        "caption_size":   int(data.get("caption_size", 14)),
        "caption_color":  str(data.get("caption_color", "#FFFFFF")),
        "caption_bg":     int(data.get("caption_bg", 60)),
        "remove_silence": bool(data.get("remove_silence",True)),
    }
    job_write(job_id, {"id":job_id,"status":"running","progress":0,
                       "message":"Iniciando pipeline...","stage":"init","clips":[]})
    threading.Thread(target=run_pipeline, args=(job_id,url,settings), daemon=True).start()
    return jsonify({"job_id":job_id})

def _build_settings(data):
    """Extrai settings de request JSON ou form."""
    def g(k, d): return data.get(k, d)
    return {
        "num_clips":      int(g("num_clips",5)),
        "clip_duration":  int(g("clip_duration",60)),
        "min_score":      int(g("min_score",30)),
        "aspect":         str(g("aspect","9:16")),
        "crf":            str(g("crf","20")),
        "captions":       str(g("captions","true")).lower() not in ("false","0"),
        "caption_style":  int(g("caption_style",2)),
        "caption_size":   int(g("caption_size",14)),
        "caption_color":  str(g("caption_color","#FFFFFF")),
        "caption_bg":     int(g("caption_bg",60)),
        "remove_silence": str(g("remove_silence","true")).lower() not in ("false","0"),
        "face_tracking":  str(g("face_tracking","true")).lower() not in ("false","0"),
        "crop_x": float(g("crop_x",0)), "crop_y": float(g("crop_y",0)),
        "crop_w": float(g("crop_w",1)), "crop_h": float(g("crop_h",1)),
    }


def run_pipeline_from_file(job_id, video_path, settings):
    """Pipeline a partir de arquivo já enviado pelo usuário."""
    try:
        if is_cancelled(job_id): return
        job_update(job_id, progress=15, message="🔧 Preparando vídeo...", stage="download")
        video_path = remux_video(video_path, job_id)

        job_update(job_id, progress=20, message="🔊 Extraindo áudio...", stage="download")
        audio_path = extract_audio(video_path, job_id)
        af = Path(audio_path)
        if not af.exists() or af.stat().st_size < 1000:
            return job_update(job_id, status="error", message="❌ Áudio inválido.")

        job_update(job_id, progress=30, message="🎙️ Transcrevendo...", stage="transcribe",
                   audio_size_kb=round(af.stat().st_size/1024, 1))
        transcript, raw_segments, words_data, err = transcribe_assemblyai(audio_path, job_id)
        if not transcript:
            return job_update(job_id, status="error", message=f"❌ Transcrição falhou: {err}")

        job_update(job_id, progress=50, message="🎵 Analisando energia...", stage="analyze")
        energy_data = extract_audio_energy(audio_path, job_id)

        scored = None
        if ANTHROPIC_KEY:
            job_update(job_id, progress=58, message="🤖 Claude analisando...", stage="analyze")
            scored, _ = analyze_with_claude(transcript, words_data, energy_data, settings, job_id)

        if not scored:
            segments = analyze_transcript(transcript, raw_segments, settings)
            if not segments:
                return job_update(job_id, status="error", message="❌ Nenhum segmento identificado.")
            for seg in segments:
                seg["energy_avg"], seg["energy_peak_val"] = energy_for_segment(
                    energy_data, seg["start"], seg["end"])
            scored = score_clips_opusclip(segments, transcript)

        job_update(job_id, scored_count=len(scored),
                   top_score=scored[0]["score"] if scored else 0)
        top_clips = pick_top_clips(scored, settings)

        job_update(job_id, progress=86, message="⚡ Renderizando clipes...", stage="render")
        output_clips = cut_clips(video_path, top_clips, job_id, settings, words_data)
        if not output_clips:
            return job_update(job_id, status="error", message="❌ Render falhou.")
        job_update(job_id, status="done", progress=100,
                   message=f"✅ {len(output_clips)} clipes gerados!", clips=output_clips)
    except Exception as e:
        job_update(job_id, status="error", message=f"❌ {type(e).__name__}: {e}")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Upload direto de MP4 — solução para vídeos geobloqueados."""
    if not AAI_KEY:
        return jsonify({"error":"❌ Chave AssemblyAI não configurada."}), 400
    if "file" not in request.files:
        return jsonify({"error":"Nenhum arquivo enviado."}), 400
    f = request.files["file"]
    if not (f.filename or "").lower().endswith((".mp4",".mov",".avi",".mkv",".webm")):
        return jsonify({"error":"Formato inválido. Use MP4, MOV, AVI, MKV ou WebM."}), 400
    job_id    = str(uuid.uuid4())[:8]
    out_dir   = UPLOADS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path  = out_dir / "video.mp4"
    f.save(str(vid_path))
    if not vid_path.exists() or vid_path.stat().st_size < 10000:
        return jsonify({"error":"Arquivo inválido ou muito pequeno."}), 400
    settings = _build_settings(request.form)
    job_write(job_id, {"id":job_id,"status":"running","progress":0,
                       "message":"📁 Arquivo recebido, processando...","stage":"init","clips":[]})
    threading.Thread(target=run_pipeline_from_file,
                     args=(job_id, str(vid_path), settings), daemon=True).start()
    return jsonify({"job_id": job_id})


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

@app.route("/proxy-status")
def proxy_status():
    return jsonify({
        "proxy_configured": bool(YTDLP_PROXY),
        "proxy": (YTDLP_PROXY[:25]+"...") if YTDLP_PROXY else "nao configurado",
    })

@app.route("/cookies-status")
def cookies_status():
    has_file = COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 100
    count = 0
    if has_file:
        with open(COOKIES_FILE) as f:
            count = sum(1 for l in f if l.strip() and not l.startswith("#"))
    return jsonify({
        "env_var_present": bool(os.environ.get("YOUTUBE_COOKIES","")),
        "file_exists": has_file,
        "cookies_count": count,
    })

@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    job = job_read(job_id)
    if not job:
        return jsonify({"error": "Job não encontrado"}), 404
    if job.get("status") in ("done", "error", "cancelled"):
        return jsonify({"status": job["status"], "message": "Job já finalizado."})
    cancel_job(job_id)
    return jsonify({"ok": True, "message": "⛔ Cancelamento solicitado."})

@app.route("/jobs/recent")
def recent_jobs():
    """Lista os últimos 20 jobs para o histórico de downloads."""
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        try:
            j = json.loads(p.read_text())
            jobs.append({
                "id":       j.get("id",""),
                "status":   j.get("status",""),
                "message":  j.get("message",""),
                "progress": j.get("progress",0),
                "clips":    j.get("clips",[]),
                "top_score":j.get("top_score",0),
                "mtime":    int(p.stat().st_mtime),
            })
        except: pass
    return jsonify(jobs)

@app.route("/health")
def health():
    return jsonify({"status":"ok","domain":"autoclip.up.railway.app"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=False)
