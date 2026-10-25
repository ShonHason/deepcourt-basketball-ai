from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "data/output/detection_core_hard_frames_trusted_appearance"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepCourt Labeler</title>
  <style>
    html, body { height: 100%; margin: 0; overflow: hidden; background: #0d0d0d; color: #f4f4f5; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-size: 13px; }
    * { box-sizing: border-box; }

    .app { display: grid; grid-template-columns: 170px 1fr; height: 100%; }

    /* ── Sidebar ── */
    aside { display: flex; flex-direction: column; border-right: 1px solid #1e1e1e; background: #111; overflow: hidden; }
    .side-head { padding: 8px 10px; border-bottom: 1px solid #1e1e1e; flex-shrink: 0; }
    .side-head h1 { font-size: 12px; font-weight: 700; margin: 0 0 5px; color: #7adb8f; letter-spacing: .03em; }
    .progress-bar { height: 3px; background: #1e1e1e; border-radius: 2px; margin-bottom: 4px; }
    .progress-fill { height: 100%; background: #7adb8f; border-radius: 2px; transition: width .3s; }
    .stats { font-size: 11px; color: #444; }
    .stats b { color: #999; }

    .classes { padding: 6px 8px; border-bottom: 1px solid #1e1e1e; flex-shrink: 0; display: grid; gap: 3px; }
    .class-btn {
      display: flex; justify-content: space-between; align-items: center;
      padding: 5px 8px; border-radius: 5px; border: 1.5px solid #252525; background: #181818;
      font-size: 12px; cursor: pointer; color: #bbb; transition: border-color .15s;
    }
    .class-btn:hover { border-color: #444; }
    .class-btn.active { border-color: #7adb8f; background: #0a2218; color: #7adb8f; }
    .class-btn .kbd { font-size: 10px; background: #252525; border-radius: 2px; padding: 1px 4px; color: #666; }

    .img-list { flex: 1; overflow-y: auto; padding: 4px 6px; }
    .img-list::-webkit-scrollbar { width: 4px; }
    .img-list::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
    .img-item {
      display: flex; align-items: center; gap: 4px;
      padding: 3px 5px; border-radius: 4px; cursor: pointer; font-size: 10px;
      color: #444; white-space: nowrap; overflow: hidden;
    }
    .img-item:hover { background: #191919; color: #999; }
    .img-item.current { background: #0a2218; color: #7adb8f; }
    .img-item.saved   { color: #2d2d2d; }
    .img-item .dot    { width: 5px; height: 5px; border-radius: 50%; background: #2a2a2a; flex-shrink: 0; }
    .img-item.current .dot { background: #7adb8f; }
    .img-item.saved   .dot { background: #14532d; }
    .itag { font-size: 9px; padding: 0 3px; border-radius: 2px; flex-shrink: 0; line-height: 14px; }
    .itag.dup  { background: #3b1a00; color: #fb923c; }
    .itag.huge { background: #1e1e3b; color: #818cf8; }

    /* ── Main ── */
    main { display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

    /* Single-row toolbar — never wraps */
    .toolbar {
      display: flex; align-items: center; gap: 4px;
      padding: 5px 8px; border-bottom: 1px solid #1e1e1e; background: #0f0f0f;
      flex-shrink: 0; overflow-x: auto; min-height: 40px;
    }
    .toolbar::-webkit-scrollbar { height: 0; }

    .btn {
      border: 1px solid #2c2c2c; background: #1a1a1a; color: #bbb;
      height: 28px; border-radius: 5px; padding: 0 9px; font: inherit; font-size: 12px;
      cursor: pointer; white-space: nowrap; flex-shrink: 0; transition: background .1s;
    }
    .btn:hover { background: #242424; border-color: #444; color: #eee; }
    .btn.primary { background: #0a3320; border-color: #1a7a4a; color: #5de89a; font-weight: 600; }
    .btn.primary:hover { background: #0d4028; }
    .btn.orange  { background: #271200; border-color: #9a3a10; color: #f4955a; }
    .btn.orange:hover  { background: #341800; }
    .btn.red     { background: #1e0606; border-color: #7a1a1a; color: #f08080; }
    .btn.red:hover     { background: #2a0808; }
    .btn.yellow  { background: #180f00; border-color: #806010; color: #e0b040; }
    .btn.yellow:hover  { background: #201400; }

    .sep { width: 1px; height: 18px; background: #222; flex-shrink: 0; margin: 0 2px; }
    .ctr { font-size: 11px; color: #555; white-space: nowrap; flex-shrink: 0; padding: 0 4px; }
    .status-txt { font-size: 11px; color: #444; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-left: 4px; }

    /* ── Canvas stage ── */
    .stage { flex: 1; position: relative; overflow: hidden; background: #080808; }

    /* canvas-wrap is positioned purely by JS for clean centering+zoom */
    .canvas-wrap { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
    #frameImage  { display: block; }
    canvas       { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; display: block; }

    /* Overlays (all fixed to stage, not canvas-wrap) */
    .ov {
      position: absolute; z-index: 10; pointer-events: none;
      background: rgba(8,8,8,.82); border: 1px solid #2a2a2a; border-radius: 5px;
      padding: 4px 10px; font-size: 12px;
    }
    .ov-tl { top: 10px; left: 10px; display: flex; align-items: center; gap: 7px; }
    .ov-tr { top: 10px; right: 10px; font-size: 13px; font-weight: 600; letter-spacing: .02em; }
    .ov-tr.dup  { border-color: #c05010; color: #fb923c; }
    .ov-tr.huge { border-color: #5050c0; color: #a0a0f8; }
    .ov-bl { bottom: 10px; left: 10px; font-size: 11px; color: #555; }
    .ov-br { bottom: 10px; right: 10px; font-size: 11px; color: #444; }

    .ptag { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; }
    .ptag.dup  { background: #3b1a00; color: #fb923c; }
    .ptag.huge { background: #1e1e3b; color: #818cf8; }

    .tip { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); z-index: 10;
      pointer-events: none; background: rgba(8,8,8,.75); border: 1px solid #1e1e1e;
      border-radius: 5px; padding: 4px 14px; font-size: 10px; color: #444; white-space: nowrap; }
    .tip b { color: #7adb8f; }
  </style>
</head>
<body>
<div class="app">

  <!-- ── Sidebar ── -->
  <aside>
    <div class="side-head">
      <h1>DeepCourt Labeler</h1>
      <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="classes" id="classes"></div>
    <div class="img-list" id="imageList"></div>
  </aside>

  <!-- ── Main ── -->
  <main>
    <div class="toolbar">
      <button class="btn" id="prevBtn">◀</button>
      <button class="btn primary" id="saveNextBtn">✓ Save &amp; Next &nbsp;↵</button>
      <button class="btn" id="nextBtn">▶</button>
      <div class="sep"></div>
      <button class="btn orange" id="keepBestBtn">K &nbsp;Keep Best Ball</button>
      <button class="btn red"    id="clearBallsBtn">C &nbsp;Clear Balls</button>
      <button class="btn red"    id="deleteBtn">Del &nbsp;Delete Box</button>
      <div class="sep"></div>
      <button class="btn yellow" id="emptyBtn">E &nbsp;No Ball</button>
      <div class="sep"></div>
      <span class="ctr" id="ctr">–/–</span>
      <span class="status-txt" id="status">Loading…</span>
    </div>

    <div class="stage" id="stage">
      <div class="canvas-wrap" id="canvasWrap">
        <img id="frameImage" alt="" />
        <canvas id="canvas"></canvas>
      </div>

      <!-- overlays fixed to stage viewport -->
      <div class="ov ov-tl" id="ovTL"></div>
      <div class="ov ov-tr" id="ovTR" style="display:none"></div>
      <div class="ov ov-bl" id="ovBL">no boxes</div>
      <div class="ov ov-br" id="ovBR">–%</div>
      <div class="tip"><b>K</b>=keep best &nbsp;<b>C</b>=clear balls &nbsp;<b>E</b>=no ball &nbsp;<b>Del</b>=delete &nbsp;<b>Scroll</b>=zoom &nbsp;<b>0</b>=fit &nbsp;<b>1/2/3</b>=class</div>
    </div>
  </main>
</div>

<script>
  const COLORS = ["#f59e0b","#38bdf8","#a78bfa","#22c55e","#fb7185","#f472b6"];
  let imageEl  = document.getElementById("frameImage");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d");
  const wrap   = document.getElementById("canvasWrap");
  const stage  = document.getElementById("stage");

  let images = [], classes = [], index = 0, boxes = [], selected = -1;
  let activeClass = 0, drawing = false, start = null, draft = null, dirty = false, loadToken = 0;
  let zoom = 1, panX = 0, panY = 0;

  const $  = id => document.getElementById(id);
  const setStatus = t => $("status").textContent = t;
  const imageName = () => images[index]?.name || "";

  function problemTag(name) {
    if (name.startsWith("dup_"))  return "dup";
    if (name.startsWith("huge_")) return "huge";
    return "";
  }

  // ── Zoom / fit ────────────────────────────────────────────────────────────
  function computeFitZoom() {
    if (!imageEl.naturalWidth) return 1;
    const sr = stage.getBoundingClientRect();
    return Math.min((sr.width - 8) / imageEl.naturalWidth, (sr.height - 8) / imageEl.naturalHeight);
  }

  function applyZoom() {
    if (!imageEl.naturalWidth) return;
    const sr = stage.getBoundingClientRect();
    const tx = (sr.width  - imageEl.naturalWidth  * zoom) / 2 + panX;
    const ty = (sr.height - imageEl.naturalHeight * zoom) / 2 + panY;
    wrap.style.transform = `translate(${tx}px,${ty}px) scale(${zoom})`;
    $("ovBR").textContent = Math.round(zoom * 100) + "%";
  }

  // ── Overlays ──────────────────────────────────────────────────────────────
  function updateOverlays() {
    const tag = problemTag(imageName());

    // top-left: frame counter + problem tag
    const tagHtml = tag ? `<span class="ptag ${tag}">${tag === "dup" ? "DUP BALLS" : "HUGE BOX"}</span>` : "";
    $("ovTL").innerHTML = `<span style="color:#555">${index+1}/${images.length}</span>${tagHtml}`;

    // top-right: action hint
    const hint = $("ovTR");
    if (tag === "dup") {
      hint.className = "ov ov-tr dup";
      hint.textContent = "→  K  then  Enter";
      hint.style.display = "";
    } else if (tag === "huge") {
      hint.className = "ov ov-tr huge";
      hint.textContent = "→  C  then draw  then  Enter";
      hint.style.display = "";
    } else {
      hint.style.display = "none";
    }

    $("ctr").textContent = `${index+1} / ${images.length}`;
  }

  function updateBoxOverlay() {
    if (!boxes.length) { $("ovBL").textContent = "no boxes"; return; }
    const counts = {};
    boxes.forEach(b => {
      const n = classes[b.class_id] || `cls${b.class_id}`;
      counts[n] = (counts[n] || 0) + 1;
    });
    $("ovBL").textContent = Object.entries(counts).map(([k,v]) => `${v}× ${k}`).join("  ");
  }

  // ── Canvas drawing ────────────────────────────────────────────────────────
  function draw() {
    if (!imageEl.complete || !imageEl.naturalWidth) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    [...boxes, draft].filter(Boolean).forEach((box, i) => {
      const isSel = i === selected;
      const color = COLORS[box.class_id % COLORS.length];
      ctx.lineWidth   = isSel ? 3 : 2;
      ctx.strokeStyle = color;
      if (isSel) { ctx.fillStyle = color + "22"; ctx.fillRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1); }
      ctx.strokeRect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1);
      const label = classes[box.class_id] || `cls${box.class_id}`;
      ctx.font = "bold 13px system-ui";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      const ly = Math.max(0, box.y1 - 18);
      ctx.fillRect(box.x1, ly, tw, 18);
      ctx.fillStyle = "#000"; ctx.fillText(label, box.x1+4, ly+13);
      const bw = Math.round(box.x2-box.x1), bh = Math.round(box.y2-box.y1);
      ctx.fillStyle = color+"cc"; ctx.font = "10px system-ui";
      ctx.fillText(`${bw}×${bh}`, box.x1+2, box.y2-3);
    });
    updateBoxOverlay();
  }

  // ── Sidebar ───────────────────────────────────────────────────────────────
  function renderClasses() {
    $("classes").innerHTML = "";
    classes.forEach((name, i) => {
      const btn = document.createElement("button");
      btn.className = "class-btn" + (i === activeClass ? " active" : "");
      const color = COLORS[i % COLORS.length];
      btn.innerHTML = `<span style="display:flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></span>${name}</span><span class="kbd">${i+1}</span>`;
      btn.onclick = () => { activeClass = i; renderClasses(); };
      $("classes").appendChild(btn);
    });
  }

  function renderImageList() {
    const saved = images.filter(im => im.labeled).length;
    $("progressFill").style.width = (images.length ? saved/images.length*100 : 0) + "%";
    $("stats").innerHTML = `<b>${saved}</b>/${images.length} done · <b>${images.length-saved}</b> left`;
    $("imageList").innerHTML = "";
    images.forEach((item, i) => {
      const row = document.createElement("div");
      const tag = problemTag(item.name);
      row.className = "img-item" + (i===index ? " current" : item.labeled ? " saved" : "");
      const tagHtml = tag ? `<span class="itag ${tag}">${tag}</span>` : "";
      const short = item.name.replace(/_jpg\.rf\.[a-f0-9]+\.jpg$/, "").replace(/_jpg$/, "").slice(0,22);
      row.innerHTML = `<span class="dot"></span>${tagHtml}<span style="flex:1;overflow:hidden;text-overflow:ellipsis" title="${item.name}">${short}</span>`;
      row.onclick = () => navigateToIndex(i);
      $("imageList").appendChild(row);
      if (i === index) row.scrollIntoView({ block: "nearest" });
    });
  }

  // ── Canvas interaction ────────────────────────────────────────────────────
  function canvasPoint(evt) {
    const r = canvas.getBoundingClientRect();
    return { x: (evt.clientX-r.left)*canvas.width/r.width, y: (evt.clientY-r.top)*canvas.height/r.height };
  }
  function hitBox(pt) {
    for (let i = boxes.length-1; i >= 0; i--) {
      const b = boxes[i];
      if (pt.x>=b.x1 && pt.x<=b.x2 && pt.y>=b.y1 && pt.y<=b.y2) return i;
    }
    return -1;
  }
  function normBox(a, b, cls) {
    return { class_id:cls, x1:Math.max(0,Math.min(a.x,b.x)), y1:Math.max(0,Math.min(a.y,b.y)),
             x2:Math.min(canvas.width,Math.max(a.x,b.x)), y2:Math.min(canvas.height,Math.max(a.y,b.y)) };
  }

  canvas.addEventListener("mousedown", evt => {
    const pt = canvasPoint(evt), hit = hitBox(pt);
    if (hit >= 0) { selected=hit; activeClass=boxes[hit].class_id; renderClasses(); draw(); return; }
    start=pt; drawing=true; draft={class_id:activeClass, x1:pt.x, y1:pt.y, x2:pt.x, y2:pt.y};
  });
  canvas.addEventListener("mousemove", evt => {
    if (!drawing) return; draft=normBox(start,canvasPoint(evt),activeClass); draw();
  });
  canvas.addEventListener("mouseup", evt => {
    if (!drawing) return; drawing=false;
    const box=normBox(start,canvasPoint(evt),activeClass); draft=null;
    if (Math.abs(box.x2-box.x1)>4 && Math.abs(box.y2-box.y1)>4) { boxes.push(box); selected=boxes.length-1; dirty=true; }
    draw();
  });
  canvas.addEventListener("contextmenu", evt => {
    evt.preventDefault(); const pt=canvasPoint(evt), hit=hitBox(pt);
    if (hit>=0) { selected=hit; deleteSelected(); }
  });

  stage.addEventListener("wheel", evt => {
    evt.preventDefault();
    const delta = evt.deltaY < 0 ? 1.12 : 1/1.12;
    zoom = Math.max(0.08, Math.min(10, zoom*delta));
    applyZoom();
  }, { passive: false });

  // ── API ───────────────────────────────────────────────────────────────────
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function loadImage(nextIndex) {
    const token = ++loadToken;
    index = Math.max(0, Math.min(images.length-1, nextIndex));
    selected=-1; dirty=false; panX=0; panY=0;
    updateOverlays(); renderImageList(); setStatus("Loading…");

    const name = encodeURIComponent(imageName());
    boxes=[]; canvas.width=1; canvas.height=1; ctx.clearRect(0,0,1,1);

    const fresh = imageEl.cloneNode(false);
    fresh.id="frameImage"; fresh.alt="";
    imageEl.replaceWith(fresh); imageEl=fresh;

    imageEl.onload = () => {
      if (token!==loadToken) return;
      canvas.width=imageEl.naturalWidth; canvas.height=imageEl.naturalHeight;
      zoom=computeFitZoom(); applyZoom(); draw();
      setStatus(`${index+1}/${images.length}: ${imageName()}`);
      updateOverlays();
    };
    imageEl.onerror = () => setStatus("Could not load image");
    imageEl.src = `/image/${name}?t=${Date.now()}`;

    const data = await api(`/api/labels?image=${name}`);
    if (token!==loadToken) return;
    boxes = data.boxes||[];
    if (imageEl.complete && imageEl.naturalWidth) {
      canvas.width=imageEl.naturalWidth; canvas.height=imageEl.naturalHeight;
      zoom=computeFitZoom(); applyZoom(); draw();
      setStatus(`${index+1}/${images.length}: ${imageName()}`);
    }
  }

  async function saveLabels(empty=false) {
    const result = await api("/api/labels", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ image:imageName(), boxes: empty ? [] : boxes }),
    });
    images[index].labeled=true;
    images[index].counts=result.counts||images[index].counts||{};
    dirty=false; setStatus("Saved ✓"); renderImageList();
  }

  async function saveAndNext() {
    await saveLabels(false);
    if (index < images.length-1) await loadImage(index+1); else setStatus("All done!");
  }

  async function navigateToIndex(i) {
    if (dirty) await saveLabels();
    await loadImage(i);
  }

  // ── Box actions ───────────────────────────────────────────────────────────
  function deleteSelected() {
    if (selected<0) return; boxes.splice(selected,1); selected=-1; dirty=true; draw();
  }

  function keepBestBall() {
    const balls = boxes.map((b,i)=>({i,area:(b.x2-b.x1)*(b.y2-b.y1),b})).filter(x=>x.b.class_id===0);
    if (balls.length<=1) { setStatus("Only one ball box"); return; }
    balls.sort((a,b)=>a.area-b.area);
    const keepIdx=balls[0].i;
    boxes=boxes.filter((box,i)=>box.class_id!==0||i===keepIdx);
    selected=-1; dirty=true; draw();
    setStatus(`Kept tightest ball, removed ${balls.length-1} dup(s)`);
  }

  function clearAllBalls() {
    boxes=boxes.filter(b=>b.class_id!==0); selected=-1; dirty=true; draw();
    setStatus("Cleared — draw a tight ball box");
  }

  // ── Toolbar ───────────────────────────────────────────────────────────────
  $("saveNextBtn").onclick   = () => saveAndNext().catch(e=>setStatus(e.message));
  $("prevBtn").onclick       = () => navigateToIndex(Math.max(0,index-1)).catch(e=>setStatus(e.message));
  $("nextBtn").onclick       = () => navigateToIndex(Math.min(images.length-1,index+1)).catch(e=>setStatus(e.message));
  $("deleteBtn").onclick     = deleteSelected;
  $("keepBestBtn").onclick   = keepBestBall;
  $("clearBallsBtn").onclick = clearAllBalls;
  $("emptyBtn").onclick      = () => saveLabels(true).then(()=>{ if(index<images.length-1) loadImage(index+1); else setStatus("All done!"); }).catch(e=>setStatus(e.message));

  // ── Keyboard ──────────────────────────────────────────────────────────────
  window.addEventListener("keydown", async evt => {
    if (evt.target.tagName==="INPUT") return;
    const k=evt.key;
    if (k>="1"&&k<="9") { const c=Number(k)-1; if(c<classes.length){activeClass=c;renderClasses();} }
    else if (k==="Delete"||k==="Backspace") { deleteSelected(); }
    else if (k.toLowerCase()==="s"||k==="Enter") { evt.preventDefault(); saveAndNext().catch(e=>setStatus(e.message)); }
    else if (k.toLowerCase()==="k") { keepBestBall(); }
    else if (k.toLowerCase()==="c") { clearAllBalls(); }
    else if (k.toLowerCase()==="e") { await saveLabels(true); if(index<images.length-1) loadImage(index+1); else setStatus("All done!"); }
    else if (k==="ArrowRight"||k==="ArrowDown") { evt.preventDefault(); navigateToIndex(Math.min(images.length-1,index+1)).catch(e=>setStatus(e.message)); }
    else if (k==="ArrowLeft"||k==="ArrowUp")    { evt.preventDefault(); navigateToIndex(Math.max(0,index-1)).catch(e=>setStatus(e.message)); }
    else if (k==="="||k==="+") { zoom=Math.min(10,zoom*1.2); applyZoom(); }
    else if (k==="-")           { zoom=Math.max(0.08,zoom/1.2); applyZoom(); }
    else if (k==="0")           { zoom=computeFitZoom(); panX=0; panY=0; applyZoom(); }
  });

  window.addEventListener("resize", () => { zoom=computeFitZoom(); panX=0; panY=0; applyZoom(); });

  // ── Init ──────────────────────────────────────────────────────────────────
  async function init() {
    const data = await api("/api/images");
    images=data.images; classes=data.classes;
    renderClasses();
    if (!images.length) { setStatus("No images found"); return; }
    const startIdx = parseInt(new URLSearchParams(window.location.search).get("i")||"0",10);
    await loadImage(isFinite(startIdx)?startIdx:0);
  }
  init().catch(e=>setStatus(e.message));
</script>
</body>
</html>
"""


class LabelServer(BaseHTTPRequestHandler):
    root: Path
    images_dir: Path
    labels_dir: Path
    classes: list[str]
    status_path: Path

    def log_message(self, format: str, *args) -> None:
        print(format % args)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/images":
            statuses = self.read_statuses()
            images = []
            for path in sorted(self.images_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                label_path = self.labels_dir / f"{path.stem}.txt"
                labeled = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())
                images.append({
                    "name": path.name,
                    "labeled": labeled,
                    "status": statuses.get(path.name, ""),
                    "counts": self.label_counts(path.name),
                })
            self.send_json({"images": images, "classes": self.classes})
            return

        if parsed.path == "/api/labels":
            image = parse_qs(parsed.query).get("image", [""])[0]
            self.send_json({"boxes": self.read_labels(image)})
            return

        if parsed.path.startswith("/image/"):
            name = unquote(parsed.path.removeprefix("/image/"))
            image_path = safe_child(self.images_dir, name)
            if image_path is None or not image_path.exists():
                self.send_error(404)
                return
            mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            data = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image = payload.get("image", "")
            status = payload.get("status", "")
            image_path = safe_child(self.images_dir, image)
            if image_path is None or not image_path.exists():
                self.send_json({"error": "image not found"}, status=404)
                return
            if status not in {"", "reviewed", "needs_fix"}:
                self.send_json({"error": "bad status"}, status=400)
                return
            statuses = self.read_statuses()
            if status:
                statuses[Path(image).name] = status
            else:
                statuses.pop(Path(image).name, None)
            self.write_statuses(statuses)
            self.send_json({"ok": True, "status": status})
            return

        if parsed.path != "/api/labels":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image = payload.get("image", "")
        boxes = payload.get("boxes", [])
        image_path = safe_child(self.images_dir, image)
        if image_path is None or not image_path.exists():
            self.send_json({"error": "image not found"}, status=404)
            return
        label_path = self.labels_dir / f"{Path(image).stem}.txt"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        lines = [box_to_yolo_line(box, image_path) for box in boxes]
        label_path.write_text("".join(lines), encoding="utf-8")
        self.send_json({"ok": True, "label": label_path.name, "counts": self.label_counts(image)})

    def read_labels(self, image: str) -> list[dict[str, float | int]]:
        image_path = safe_child(self.images_dir, image)
        if image_path is None or not image_path.exists():
            return []
        label_path = self.labels_dir / f"{Path(image).stem}.txt"
        if not label_path.exists():
            return []
        width, height = image_size(image_path)
        boxes = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append({
                "class_id": class_id,
                "x1": (cx - bw / 2) * width,
                "y1": (cy - bh / 2) * height,
                "x2": (cx + bw / 2) * width,
                "y2": (cy + bh / 2) * height,
            })
        return boxes

    def label_counts(self, image: str) -> dict[str, int]:
        counts = {name: 0 for name in self.classes}
        label_path = self.labels_dir / f"{Path(image).stem}.txt"
        if not label_path.exists():
            return counts
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            if 0 <= class_id < len(self.classes):
                counts[self.classes[class_id]] = counts.get(self.classes[class_id], 0) + 1
        return counts

    def read_statuses(self) -> dict[str, str]:
        if not self.status_path.exists():
            return {}
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {str(k): str(v) for k, v in payload.items()}

    def write_statuses(self, statuses: dict[str, str]) -> None:
        self.status_path.write_text(json.dumps(statuses, indent=2, sort_keys=True), encoding="utf-8")


def safe_child(parent: Path, name: str) -> Path | None:
    path = parent / Path(name).name
    try:
        path.relative_to(parent)
    except ValueError:
        return None
    return path


def image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for local labeler") from exc
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h


def box_to_yolo_line(box: dict, image_path: Path) -> str:
    width, height = image_size(image_path)
    x1 = max(0.0, min(float(width), float(box["x1"])))
    y1 = max(0.0, min(float(height), float(box["y1"])))
    x2 = max(0.0, min(float(width), float(box["x2"])))
    y2 = max(0.0, min(float(height), float(box["y2"])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    class_id = int(box["class_id"])
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def read_classes(root: Path) -> list[str]:
    classes_path = root / "classes.txt"
    if classes_path.exists():
        classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if classes:
            return classes
    return ["ball", "main_rim", "other_rim"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser YOLO labeler for DeepCourt hard frames.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder containing images/, labels/, classes.txt.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    root = args.root.resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images directory: {images_dir}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    LabelServer.root = root
    LabelServer.images_dir = images_dir
    LabelServer.labels_dir = labels_dir
    LabelServer.classes = read_classes(root)
    LabelServer.status_path = root / "review_status.json"

    server = ThreadingHTTPServer(("127.0.0.1", args.port), LabelServer)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Labeler running: {url}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
