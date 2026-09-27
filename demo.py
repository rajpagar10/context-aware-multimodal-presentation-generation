"""Local browser demo for the multimodal ingestion pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DEMO_OUTPUT = ROOT / "outputs" / "browser_demo"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Studio · Pipeline Demo</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#18251f;--muted:#718078;--paper:#f5f7f3;--line:#e3e9e2;--green:#286747;--lime:#d8f36a;--white:#fff;--orange:#d48345}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}button,input{font:inherit}.shell{max-width:1280px;margin:auto;padding:28px 44px 56px}.top{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:20px}.brand{display:flex;align-items:center;gap:12px;font:700 15px Manrope,sans-serif;letter-spacing:-.3px}.mark{width:34px;height:34px;border-radius:11px;background:var(--green);display:grid;place-items:center;color:var(--lime);font-size:19px}.top-right{font:11px 'DM Mono',monospace;color:var(--muted);letter-spacing:.7px}.hero{display:flex;justify-content:space-between;align-items:end;padding:43px 0 27px;gap:30px}.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:1.8px;text-transform:uppercase;color:var(--green)}h1{font:700 clamp(31px,4vw,48px)/1.08 Manrope,sans-serif;letter-spacing:-2px;margin:11px 0 12px}.hero p{color:var(--muted);font-size:15px;margin:0;max-width:570px;line-height:1.6}.tag{border:1px solid #d4dfd4;border-radius:99px;padding:8px 13px;color:var(--green);font:11px 'DM Mono',monospace;white-space:nowrap}.layout{display:grid;grid-template-columns:350px 1fr;gap:22px;align-items:start}.card{background:var(--white);border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 18px #213b2910}.setup{padding:22px;position:sticky;top:18px}.section-title{font:700 14px Manrope,sans-serif;margin:0 0 18px}.drop{display:grid;place-items:center;text-align:center;border:1.5px dashed #cbd8ca;border-radius:12px;background:#f8faf7;min-height:160px;padding:18px;cursor:pointer;transition:.2s}.drop:hover,.drop.over{border-color:var(--green);background:#f1f7ef}.upload-icon{font-size:24px;margin-bottom:8px}.drop strong{font-size:13px}.drop small{color:var(--muted);font-size:11px;display:block;margin-top:6px}.filename{font:11px 'DM Mono',monospace;color:var(--green);margin-top:10px;overflow-wrap:anywhere}input[type=file]{display:none}.opts{margin:19px 0 0;display:grid;gap:12px}.check{display:flex;align-items:flex-start;gap:10px;font-size:12px;color:#45564c;line-height:1.45}.check input{accent-color:var(--green);margin:2px 0 0}.run{margin-top:21px;width:100%;border:0;border-radius:10px;padding:13px 16px;background:var(--green);color:white;font:600 13px Manrope,sans-serif;cursor:pointer;transition:.2s}.run:hover{background:#1e5338;transform:translateY(-1px)}.run:disabled{opacity:.5;cursor:wait;transform:none}.hint{font-size:10px;color:var(--muted);line-height:1.5;margin:12px 0 0}.main{display:grid;gap:18px}.empty{padding:48px 24px;text-align:center;min-height:310px;display:grid;place-content:center}.empty .orb{width:56px;height:56px;border-radius:18px;background:#eff5e9;margin:0 auto 16px;display:grid;place-items:center;font-size:24px}.empty h2{font:700 17px Manrope;margin:0 0 7px}.empty p{color:var(--muted);font-size:12px;margin:0;max-width:370px;line-height:1.6}.progress{display:none;padding:22px}.progress.on{display:block}.progress-line{height:5px;background:#edf1ec;border-radius:5px;overflow:hidden;margin-top:15px}.progress-line i{display:block;height:100%;width:25%;background:var(--lime);animation:move 1s ease-in-out infinite alternate}@keyframes move{to{width:90%}}.summary{display:none}.summary.on{display:grid;gap:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:17px 18px}.stat label{display:block;text-transform:uppercase;letter-spacing:1px;font:10px 'DM Mono',monospace;color:var(--muted)}.stat strong{font:700 24px Manrope;display:block;margin-top:9px}.stat span{font-size:10px;color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{padding:20px}.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px}.panel-head h2{font:700 14px Manrope;margin:0}.pill{font:10px 'DM Mono',monospace;padding:5px 8px;border-radius:99px;background:#edf5e9;color:var(--green)}.content-list{display:grid;gap:10px;max-height:250px;overflow:auto}.content-item{padding:12px;border:1px solid var(--line);border-radius:10px}.content-meta{font:10px 'DM Mono',monospace;color:var(--muted);margin-bottom:6px}.content-text{font-size:12px;line-height:1.5;white-space:pre-wrap}.word-list{display:grid;gap:8px}.word-row{display:grid;grid-template-columns:1fr auto auto;gap:13px;align-items:center;padding:8px 0;border-bottom:1px solid #eef1ed;font-size:12px}.word-row:last-child{border:0}.word-time{font:10px 'DM Mono',monospace;color:var(--muted)}.confidence{font:10px 'DM Mono',monospace;color:var(--green)}.notice{padding:11px 12px;border-radius:9px;background:#fff6e9;color:#805428;font-size:11px;line-height:1.5;margin-top:10px}.json-card{overflow:hidden}.json-toolbar{display:flex;justify-content:space-between;align-items:center;padding:15px 18px;border-bottom:1px solid var(--line)}.json-toolbar h2{font:700 13px Manrope;margin:0}.json-toolbar button{border:1px solid var(--line);background:white;border-radius:7px;padding:7px 10px;font-size:10px;cursor:pointer}.json{padding:16px 18px;margin:0;max-height:330px;overflow:auto;background:#fbfcfa;font:10px/1.6 'DM Mono',monospace;color:#33483b;white-space:pre-wrap;word-break:break-word}.footer{display:flex;justify-content:space-between;margin-top:22px;color:#8a968d;font:10px 'DM Mono',monospace}.error{display:none;padding:14px;background:#fff0ed;border:1px solid #f2d1ca;border-radius:10px;color:#a6402f;font-size:12px;line-height:1.5}.error.on{display:block}
@media(max-width:900px){.shell{padding:20px}.layout{grid-template-columns:1fr}.setup{position:static}.two{grid-template-columns:1fr}}@media(max-width:600px){.hero{display:block}.tag{display:inline-block;margin-top:17px}.stats{grid-template-columns:repeat(2,1fr)}.top-right{display:none}}
</style>
</head>
<body><div class="shell">
<header class="top"><div class="brand"><div class="mark">◌</div> SIGNAL STUDIO <span style="color:#9aa79d;font-weight:400">/ PIPELINE DEMO</span></div><div class="top-right">LOCAL RESEARCH PROTOTYPE &nbsp;·&nbsp; TASKS 1 + 2</div></header>
<section class="hero"><div><div class="eyebrow">Context-aware multimodal presentation generation</div><h1>From source file<br>to structured insight.</h1><p>Run the ingestion pipeline and inspect how content, narration, and speech evidence are assembled into one normalized representation.</p></div><div class="tag">LIVE PIPELINE DEMO</div></section>
<div class="layout"><aside class="card setup"><h2 class="section-title">Configure a run</h2><label id="drop" class="drop" for="file"><div><div class="upload-icon">↥</div><strong>Choose a file or drop it here</strong><small>PPTX · PDF · DOCX · TXT · WAV · MP3 · MP4</small><div class="filename" id="filename"></div></div></label><input id="file" type="file" accept=".pptx,.pdf,.docx,.txt,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.mp4,.mov,.mkv,.avi,.webm,.m4v">
<div class="opts"><label class="check"><input id="narration" type="checkbox"><span>Generate narration for document or presentation content</span></label><label class="check"><input id="speech" type="checkbox"><span>Enable speech analysis (Task 2)</span></label></div>
<button id="run" class="run">Run ingestion pipeline&nbsp; →</button><p class="hint">Audio/video transcription requires FFmpeg where applicable and the configured Faster-Whisper model. The run uses the repository’s existing CLI and writes results under <code>outputs/browser_demo</code>.</p></aside>
<main class="main"><div class="error" id="error"></div><section class="card empty" id="empty"><div><div class="orb">✳</div><h2>Your pipeline output will appear here</h2><p>Choose a source file and start a run. The summary will show extracted content, speech timing, and the generated JSON artifact.</p></div></section>
<section class="card progress" id="progress"><strong style="font:700 14px Manrope">Pipeline is running…</strong><div style="font-size:11px;color:var(--muted);margin-top:6px">Extracting and normalizing your input. Audio analysis can take longer on CPU.</div><div class="progress-line"><i></i></div></section>
<div class="summary" id="summary"><section class="stats"><div class="card stat"><label>Input type</label><strong id="kind">—</strong><span>detected modality</span></div><div class="card stat"><label>Content units</label><strong id="count">0</strong><span>extracted items</span></div><div class="card stat"><label>Word timings</label><strong id="words">0</strong><span>ASR-aligned words</span></div><div class="card stat"><label>Speech analysis</label><strong id="status" style="font-size:18px">—</strong><span>Task 2 status</span></div></section>
<section class="two"><div class="card panel"><div class="panel-head"><h2>Extracted content</h2><span class="pill" id="inputname">SOURCE</span></div><div class="content-list" id="items"></div></div><div class="card panel"><div class="panel-head"><h2>Speech evidence</h2><span class="pill" id="region-count">0 regions</span></div><div class="word-list" id="word-list"></div><div id="speech-notice"></div></div></section>
<section class="card json-card"><div class="json-toolbar"><h2>Normalized representation</h2><button id="download">Download JSON ↓</button></div><pre class="json" id="json"></pre></section></div></main></div>
<footer class="footer"><span>RESEARCH PROTOTYPE · LOCAL PROCESSING</span><span>CONTENT → NARRATION → SPEECH EVIDENCE</span></footer></div>
<script>
const $=s=>document.querySelector(s),file=$('#file'),drop=$('#drop');let resultData=null;
file.addEventListener('change',()=>$('#filename').textContent=file.files[0]?.name||'');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over')});drop.addEventListener('dragleave',()=>drop.classList.remove('over'));drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files.length){file.files=e.dataTransfer.files;$('#filename').textContent=file.files[0].name}});
$('#run').addEventListener('click',async()=>{const f=file.files[0];if(!f){showError('Choose an input file to start.');return}if(f.size>1024*1024*1024){showError('This demo accepts files up to 1 GB.');return}hideError();$('#empty').style.display='none';$('#summary').classList.remove('on');$('#progress').classList.add('on');$('#run').disabled=true;try{const q=new URLSearchParams({narration:$('#narration').checked?'1':'0',speech:$('#speech').checked?'1':'0'});const r=await fetch('/api/run?'+q,{method:'POST',headers:{'X-Filename':encodeURIComponent(f.name),'Content-Type':'application/octet-stream'},body:f});const data=await r.json();if(!r.ok)throw new Error(data.error||'Pipeline run failed');resultData=data;render(data)}catch(e){showError(e.message);$('#empty').style.display='grid'}finally{$('#progress').classList.remove('on');$('#run').disabled=false}});
function showError(s){const e=$('#error');e.textContent=s;e.classList.add('on')}function hideError(){$('#error').classList.remove('on')}
function render(d){const x=d.representation,a=x.speech_analysis||{};$('#kind').textContent=(x.input?.kind||'—').toUpperCase();$('#count').textContent=x.content_items?.length||0;$('#words').textContent=a.word_alignments?.length||0;$('#status').textContent=(a.status||'not_started').replace('_',' ');$('#inputname').textContent=x.input?.filename||'SOURCE';$('#region-count').textContent=`${a.speech_regions?.length||0} regions`;$('#items').innerHTML=(x.content_items||[]).map(i=>`<div class="content-item"><div class="content-meta">${esc(i.kind)} · ${i.index}</div><div class="content-text">${esc(i.text||'(No text extracted)')}</div></div>`).join('')||'<div class="content-meta">No content units returned.</div>';$('#word-list').innerHTML=(a.word_alignments||[]).slice(0,80).map(w=>`<div class="word-row"><strong>${esc(w.word)}</strong><span class="word-time">${Number(w.start_seconds).toFixed(2)}s — ${Number(w.end_seconds).toFixed(2)}s</span><span class="confidence">${w.confidence==null?'—':Number(w.confidence).toFixed(2)}</span></div>`).join('')||'<div class="content-meta">No word-level timing available for this run.</div>';const notes=[...(a.warnings||[]),...(x.warnings||[])];$('#speech-notice').innerHTML=notes.length?`<div class="notice">${notes.slice(0,3).map(esc).join('<br>')}</div>`:'';$('#json').textContent=JSON.stringify(x,null,2);$('#summary').classList.add('on')}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
$('#download').addEventListener('click',()=>{if(!resultData)return;const b=new Blob([JSON.stringify(resultData.representation,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='normalized.json';a.click();URL.revokeObjectURL(a.href)});
</script></body></html>'''


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "PipelineDemo/1.0"

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self._json_response(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_UPLOAD_BYTES:
                self._json_response(413, {"error": "Upload must be between 1 byte and 1 GB."})
                return
            filename = Path(self.headers.get("X-Filename", "input.bin")).name
            suffix = Path(filename).suffix.lower()
            safe_id = uuid.uuid4().hex
            input_path = DEMO_OUTPUT / "uploads" / f"{safe_id}{suffix}"
            output_dir = DEMO_OUTPUT / safe_id
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(self.rfile.read(size))

            query = parse_qs(urlparse(self.path).query)
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(input_path), "--output-dir", str(output_dir), "--verbose"]
            if query.get("narration") == ["1"]:
                command.append("--generate-narration")
            if query.get("speech") == ["1"]:
                command.append("--enable-speech-analysis")
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60 * 60)
            normalized_path = output_dir / "normalized.json"
            if completed.returncode != 0 or not normalized_path.exists():
                message = (completed.stderr or completed.stdout or "Pipeline did not produce normalized.json").strip()
                self._json_response(422, {"error": message[-4000:]})
                return
            representation = json.loads(normalized_path.read_text(encoding="utf-8"))
            self._json_response(200, {"representation": representation, "output": str(normalized_path.relative_to(ROOT))})
        except subprocess.TimeoutExpired:
            self._json_response(504, {"error": "Pipeline run exceeded the one-hour demo limit."})
        except Exception as exc:
            self._json_response(500, {"error": str(exc)})

    def _json_response(self, status: int, value: dict[str, object]) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[demo] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), DemoHandler)
    print("Pipeline demo is ready at http://127.0.0.1:8765")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping pipeline demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
