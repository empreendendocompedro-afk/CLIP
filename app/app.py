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

        # 3. Remuxar vídeo para garantir arquivo limpo e seekável
        job_update(job_id, progress=15, message="🔧 Preparando vídeo...", stage="download")
        video_path = remux_video(video_path, job_id)

        # 4. Áudio
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

        # 5. Extrair energia de áudio
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
            "--merge-output-format", "mp4", "-o", tmpl]

    # Adiciona cookies se disponível
    if COOKIES_FILE.exists():
        base += ["--cookies", str(COOKIES_FILE)]

    # Flags Deno para resolver n-challenge
    ejs_flags = ["--remote-components", "ejs:github", "--js-runtimes", "deno"]

    # Formato de melhor qualidade — vídeo e áudio separados mergeados em MP4
    # Ordem de preferência: 1080p VP9/H264 + opus/aac, fallback para 720p, fallback para best
    best_fmt = (
        "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/"   # H264+AAC (melhor compatibilidade)
        "bestvideo[height<=1080][vcodec^=vp9]+bestaudio/"                   # VP9+opus
        "bestvideo[height<=1080]+bestaudio/"                                # qualquer codec 1080p
        "bestvideo[height<=720]+bestaudio/"                                 # 720p fallback
        "best[height<=1080]/best"                                           # último recurso
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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
    ], capture_output=True, text=True, timeout=300)

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

    timed_transcript = "\n".join(timed_lines[:300])  # máximo 300 linhas (~50min de vídeo)

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


def extract_audio_energy(audio_path, job_id):
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
                    if val > -100:
                        energy.append({"second": second, "rms_db": val})
                    second += 1
                except:
                    pass
        if not energy:
            return []
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
    pts = [e["rms_norm"] for e in energy_data
           if start_sec <= e["second"] <= end_sec]
    if not pts:
        return 0.5, 0.5
    return round(sum(pts)/len(pts), 3), round(max(pts), 3)


def score_to_grade(score):
    if score >= 75: return "🔥 Viral"
    if score >= 55: return "⚡ Alto"
    if score >= 35: return "👍 Médio"
    return "📉 Baixo"


def analyze_with_claude(transcript, words_data, energy_data, settings, job_id):
    if not ANTHROPIC_KEY:
        return None, "ANTHROPIC_API_KEY nao configurada"
    target  = settings.get("clip_duration", 60)
    num     = settings.get("num_clips", 5)
    min_dur = max(15, target - 25)
    max_dur = target + 30

    energy_summary = ""
    if energy_data:
        peaks = []
        for i in range(0, len(energy_data), 5):
            blk = energy_data[i:i+5]
            avg = sum(b["rms_norm"] for b in blk) / len(blk)
            t   = blk[0]["second"]
            if avg > 0.65:
                peaks.append(f"{t}s({avg:.2f})")
        if peaks:
            energy_summary = "\nPICOS DE ENERGIA (alta intensidade vocal): " + ", ".join(peaks[:30])

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

    timed_transcript = "\n".join(timed_lines[:300])

    prompt = (
        "Voce e um especialista em criacao de conteudo viral para redes sociais.\n"
        f"Analise a transcricao e identifique exatamente {num} clipes virais.\n\n"
        "REGRAS CRITICAS:\n"
        "1. NUNCA corte no meio de um raciocinio, frase ou argumento\n"
        "2. Inicio e fim SEMANTICO — comeca quando uma ideia comeca, termina quando termina\n"
        f"3. Duracao alvo: {target}s (minimo {min_dur}s, maximo {max_dur}s)\n"
        "4. Priorize: picos de energia emocional > viradas/surpresas > perguntas+respostas > raciocinio completo\n"
        "5. Clipes nao podem se sobrepor\n"
        f"{energy_summary}\n\n"
        "TRANSCRICAO COM TIMESTAMPS:\n"
        f"{timed_transcript}\n\n"
        "Responda APENAS com JSON valido (sem markdown, sem explicacao):\n"
        '{"clips":[{"start":0.0,"end":0.0,"hook":"frase","score":0,'
        '"reasoning":"motivo","energy_peak":false,"has_question_answer":false,"has_turning_point":false}]}'
    )

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
            return None, f"Claude HTTP {resp.status_code}: {resp.text[:200]}"

        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        data  = json.loads(raw)
        clips = data.get("clips", [])
        if not clips:
            return None, "Claude nao retornou clipes"

        segments = []
        for c in clips:
            start = float(c.get("start", 0))
            end   = float(c.get("end",   0))
            dur   = end - start
            if dur < 10:
                continue
            seg_words = ([w["text"] for w in words_data
                          if start <= w.get("start",0)/1000 <= end + 0.5]
                         if words_data else [])
            text  = " ".join(seg_words) or c.get("hook", "")
            sc    = min(99, int(c.get("score", 50)))
            e_avg, e_peak = energy_for_segment(energy_data, start, end)
            segments.append({
                "start": start, "end": end, "text": text,
                "duration": round(dur, 1), "headline": c.get("hook", ""),
                "score": sc, "grade": score_to_grade(sc),
                "hook":  min(25, int(sc * 0.28)),
                "flow":  min(25, int(sc * 0.26)),
                "value": min(25, int(sc * 0.25)),
                "trend": min(25, int(sc * 0.21)),
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
        return None, f"JSON invalido: {e}"
    except Exception as e:
        return None, f"Erro: {type(e).__name__}: {e}"


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

        # Tentativa 1b: janela mais larga — combina segmentos até atingir duração mínima
        deduped2 = []
        min_abs  = max(20, target - 30)
        max_abs  = target + 40
        for i in range(len(raw_segments)):
            best = None
            for j in range(i, len(raw_segments)):
                end = raw_segments[j]["end"]
                dur = end - raw_segments[i]["start"]
                if dur < min_abs: continue
                if dur > max_abs: break
                best = {
                    "start":    raw_segments[i]["start"], "end": end,
                    "text":     " ".join(s["text"] for s in raw_segments[i:j+1]),
                    "duration": round(dur, 1),
                    "headline": raw_segments[i].get("headline", ""),
                }
            if best:
                # Só adiciona se não sobrepõe com nenhum já aceito
                if not any(best["start"] < d["end"] and best["end"] > d["start"]
                           for d in deduped2):
                    deduped2.append(best)
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

def cut_clips(video_path, clips, job_id, settings, words_data):
    out_dir      = OUTPUTS / job_id
    out_dir.mkdir(exist_ok=True)
    aspect       = settings.get("aspect", "9:16")
    crf          = str(settings.get("crf", "20"))
    use_captions = settings.get("captions", True)
    use_silence  = settings.get("remove_silence", True)
    results      = []

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

        # Verificar se o usuário ajustou o crop (tolerância de 1%)
        user_cropped = not (cx < 0.01 and cy < 0.01 and cw > 0.98 and ch > 0.98)

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
                # Target: múltiplo de 2, máximo 720x1280
                target_w = min(720,  auto_cw * 2); target_w -= target_w % 2
                target_h = min(1280, orig_h  * 2); target_h -= target_h % 2
                vf += [f"crop={auto_cw}:{orig_h}:(iw-{auto_cw})/2:0",
                       f"scale={target_w}:{target_h}:flags=lanczos",
                       "setsar=1"]          # ← força pixels quadrados
            elif aspect == "1:1":
                sq = min(orig_w, orig_h)
                target_sq = min(720, sq * 2); target_sq -= target_sq % 2
                vf += [f"crop={sq}:{sq}:(iw-{sq})/2:(ih-{sq})/2",
                       f"scale={target_sq}:{target_sq}:flags=lanczos",
                       "setsar=1"]
            else:
                target_w = min(1280, orig_w * 2); target_w -= target_w % 2
                target_h = min(720,  orig_h * 2); target_h -= target_h % 2
                vf += [f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease",
                       f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
                       "setsar=1"]

        # PASSO 1: Cortar segmento
        cmd1 = (["ffmpeg","-y",
                 "-ss",str(cs),"-i",video_path,"-t",str(duration),
                 "-c:v","libx264",
                 "-preset","faster",     # equilíbrio RAM vs qualidade
                 "-crf",crf,
                 "-profile:v","main",    # main: melhor qualidade que baseline
                 "-level","3.1",
                 "-pix_fmt","yuv420p",
                 "-threads","2",
                 "-c:a","aac","-b:a","128k","-ar","44100",
                 ]
                + (["-vf", ",".join(vf)] if vf else [])
                + ["-movflags","+faststart", str(tmp_path)])
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
        full_log = r1.stdout + r1.stderr
        # Captura início (erro) + fim (progresso) do log
        log_head = full_log[:600]
        log_tail = full_log[-200:]
        job_update(job_id, **{f"cut_log_{i}": log_head + "\n---\n" + log_tail,
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
            font_sz  = "22" if aspect == "9:16" else "18"
            margin_v = "60" if aspect == "9:16" else "40"
            srt_esc  = str(srt_path).replace(":", "\\:")
            style = (f"subtitles={srt_esc}:force_style="
                     f"'FontName=Arial,FontSize={font_sz},Bold=1,"
                     "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                     "BackColour=&H80000000,Outline=2,Shadow=1,"
                     f"MarginV={margin_v},Alignment=2'")
            r4 = subprocess.run([
                "ffmpeg","-y","-i",working,"-vf",style,
                "-c:v","libx264","-preset","ultrafast","-crf",crf,
                "-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
                "-threads","1","-bufsize","512k",
                "-c:a","copy","-movflags","+faststart", str(out_path)
            ], capture_output=True, text=True, timeout=600)
            job_update(job_id, **{f"sub_log_{i}": (r4.stdout+r4.stderr)[-400:]})
            if not out_path.exists() or out_path.stat().st_size < 1000:
                subprocess.run(["cp", working, str(out_path)], capture_output=True)
                job_update(job_id, **{f"clip_{i}_status": "sem legenda (sub falhou)"})
            else:
                job_update(job_id, **{f"clip_{i}_status": "OK com legenda"})
        else:
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
