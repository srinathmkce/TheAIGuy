import streamlit as st
import streamlit.components.v1 as components

BACKEND_WS = "ws://localhost:8000/ws/transcribe"

st.set_page_config(page_title="Nemotron Live Transcription", layout="wide")
st.title("Nemotron Live Speech Transcription")
st.caption("Powered by nvidia/nemotron-speech-streaming-en-0.6b")

components.html(
    f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Source Sans Pro", sans-serif;
    background: transparent;
    padding: 8px 4px;
    color: #31333f;
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }}
  .btn {{
    padding: 8px 18px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: background 0.2s;
  }}
  .btn-record  {{ background: #ff4b4b; color: #fff; }}
  .btn-record:hover  {{ background: #e03e3e; }}
  .btn-record.active {{ background: #1c1c1c; color: #fff; }}
  .btn-record.active:hover {{ background: #333; }}
  .btn-secondary {{ background: #e6e6e6; color: #31333f; }}
  .btn-secondary:hover {{ background: #d1d1d1; }}
  .status {{
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
    min-height: 18px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .dot {{
    width: 9px; height: 9px;
    background: #ff4b4b;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1s infinite;
  }}
  @keyframes blink {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.2 }} }}
  .transcript-box {{
    min-height: 220px;
    max-height: 380px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
    background: #fafafa;
    word-break: break-word;
  }}
  .cursor {{
    display: inline-block;
    width: 2px; height: 1em;
    background: #ff4b4b;
    animation: blink 0.8s infinite;
    vertical-align: text-bottom;
    margin-left: 1px;
  }}
</style>
</head>
<body>

<div class="controls">
  <button id="recordBtn" class="btn btn-record">🎙 Start Recording</button>
  <button id="clearBtn" class="btn btn-secondary">Clear</button>
  <button id="downloadBtn" class="btn btn-secondary">⬇ Download</button>
</div>

<div id="status" class="status">Ready — click the microphone to start.</div>

<div id="transcript" class="transcript-box"></div>

<script>
  const WS_URL          = "{BACKEND_WS}";
  const TARGET_SR       = 16000;
  const CHUNK_SECONDS   = 2;           // send a chunk every 2 s while recording
  const BUFFER_LIMIT    = TARGET_SR * CHUNK_SECONDS;

  let audioCtx, mediaStream, scriptNode, ws;
  let isRecording = false;
  let nativeSR    = 48000;
  let rawBuffer   = [];                // raw PCM at native sample rate

  const recordBtn    = document.getElementById("recordBtn");
  const clearBtn     = document.getElementById("clearBtn");
  const downloadBtn  = document.getElementById("downloadBtn");
  const statusEl     = document.getElementById("status");
  const transcriptEl = document.getElementById("transcript");
  let cursorEl       = null;

  // ── helpers ──────────────────────────────────────────────────────────────

  function setStatus(html) {{ statusEl.innerHTML = html; }}

  function addCursor() {{
    removeCursor();
    cursorEl = document.createElement("span");
    cursorEl.className = "cursor";
    transcriptEl.appendChild(cursorEl);
  }}

  function removeCursor() {{
    if (cursorEl) {{ cursorEl.remove(); cursorEl = null; }}
  }}

  function appendText(text) {{
    removeCursor();
    transcriptEl.insertAdjacentText("beforeend", text);
    addCursor();
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }}

  // Simple decimation downsample from nativeSR → TARGET_SR
  function downsample(samples) {{
    if (nativeSR === TARGET_SR) return samples;
    const ratio     = nativeSR / TARGET_SR;
    const outLen    = Math.round(samples.length / ratio);
    const out       = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {{
      out[i] = samples[Math.round(i * ratio)];
    }}
    return out;
  }}

  let chunksSent = 0;

  function sendChunk(nativePCM) {{
    if (!ws || ws.readyState !== WebSocket.OPEN) {{
      console.warn("sendChunk: WS not open, state =", ws ? ws.readyState : "null");
      return;
    }}
    const pcm16k = downsample(new Float32Array(nativePCM));
    ws.send(pcm16k.buffer);
    chunksSent++;
    console.log(`[audio] sent chunk ${{chunksSent}}: ${{pcm16k.length}} samples @ ${{TARGET_SR}}Hz (${{(pcm16k.length/TARGET_SR).toFixed(2)}}s)`);
    setStatus(`<span class="dot"></span> Recording… (${{chunksSent}} chunk${{chunksSent !== 1 ? "s" : ""}} sent)`);
  }}

  // ── record toggle ─────────────────────────────────────────────────────────

  recordBtn.addEventListener("click", () => {{
    isRecording ? stopRecording() : startRecording();
  }});

  async function startRecording() {{
    try {{
      mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true, video: false }});
      console.log("[audio] getUserMedia OK");
    }} catch (err) {{
      setStatus("Microphone access denied: " + err.message);
      console.error("[audio] getUserMedia failed:", err);
      return;
    }}

    audioCtx = new AudioContext();
    await audioCtx.resume();   // ensure AudioContext is running (autoplay policy)
    nativeSR = audioCtx.sampleRate;
    console.log(`[audio] AudioContext sample rate: ${{nativeSR}} Hz`);

    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {{
      console.log("[ws] connected to", WS_URL);
      setStatus('<span class="dot"></span> Recording &amp; transcribing in real time…');
    }};

    ws.onmessage = (evt) => {{
      const text = evt.data;
      console.log("[ws] received token:", JSON.stringify(text));
      if (text === "\\n") {{
        const last = transcriptEl.textContent.slice(-1);
        if (last && last !== " ") appendText(" ");
      }} else {{
        appendText(text);
      }}
    }};

    ws.onerror = (e) => {{
      console.error("[ws] error:", e);
      setStatus("WebSocket error — is the backend running on port 8000?");
    }};
    ws.onclose = (e) => {{
      console.log("[ws] closed, code:", e.code);
      if (isRecording) stopRecording();
    }};

    const source = audioCtx.createMediaStreamSource(mediaStream);
    scriptNode   = audioCtx.createScriptProcessor(4096, 1, 1);

    scriptNode.onaudioprocess = (e) => {{
      if (!isRecording) return;
      const data = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < data.length; i++) rawBuffer.push(data[i]);

      const nativeChunkSize = Math.round(nativeSR * CHUNK_SECONDS);
      while (rawBuffer.length >= nativeChunkSize) {{
        sendChunk(rawBuffer.splice(0, nativeChunkSize));
      }}
    }};

    source.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    console.log("[audio] ScriptProcessor connected");

    rawBuffer   = [];
    chunksSent  = 0;
    isRecording = true;
    recordBtn.textContent = "⏹ Stop Recording";
    recordBtn.classList.add("active");
    addCursor();
  }}

  function stopRecording() {{
    isRecording = false;

    // Flush remaining buffer
    if (rawBuffer.length > 0) {{
      console.log(`[audio] flushing remaining ${{rawBuffer.length}} native samples`);
      sendChunk(rawBuffer.splice(0));
    }} else {{
      console.log("[audio] no remaining samples to flush");
    }}

    if (scriptNode) {{ scriptNode.disconnect(); scriptNode = null; }}
    if (audioCtx)   {{ audioCtx.close();        audioCtx   = null; }}
    if (mediaStream) {{ mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }}

    // Keep WS open a few seconds so the backend can finish the last chunk
    if (ws) setTimeout(() => {{ if (ws) {{ ws.close(); ws = null; }} }}, 4000);

    recordBtn.textContent = "🎙 Start Recording";
    recordBtn.classList.remove("active");
    setStatus("Processing final chunk…");
    setTimeout(() => {{ if (!isRecording) {{ removeCursor(); setStatus("Done. Click the microphone to record again."); }} }}, 4500);
  }}

  // ── clear / download ──────────────────────────────────────────────────────

  clearBtn.addEventListener("click", () => {{
    transcriptEl.textContent = "";
    setStatus("Ready — click the microphone to start.");
  }});

  downloadBtn.addEventListener("click", () => {{
    const text = transcriptEl.textContent;
    if (!text.trim()) return;
    const blob = new Blob([text], {{ type: "text/plain" }});
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement("a"), {{ href: url, download: "transcription.txt" }});
    a.click();
    URL.revokeObjectURL(url);
  }});
</script>
</body>
</html>
""",
    height=560,
    scrolling=False,
)
