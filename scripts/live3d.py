#!/usr/bin/env python3
"""Watch a pass happen in 3-D, live, in a browser.

Everything visual in this repo was written AFTER the run: `viz.write_3d_player` emits a
self-contained file once `run_pass` returns, so a 150 s approach that goes wrong in the
first ten seconds costs you the whole 150 s before you can see any of it. `--map` gives a
live 2-D ASCII strip, which is genuinely useful and genuinely not three-dimensional.

This is the missing third thing. One process serves a page; another appends telemetry to a
JSON-lines file as the flight happens; the page polls that file's server and appends. No
build step, no CDN, no browser extension -- Python's `http.server` and one canvas, the same
camera and projection the offline player already uses, so a live view and a replay of the
same case look like the same scene.

    # terminal 1 -- start the viewer first, it works with no flight attached yet
    python3 scripts/live3d.py runs/live-iris/stream.jsonl

    # terminal 2 -- fly something, streaming as it goes
    python3 scripts/fly_live.py --profile iris --suite ground --scenario ground_on_slope \
        --stream runs/live-iris/stream.jsonl

Then open the printed URL. The flight appears as it happens; the trail grows behind it.

    python3 scripts/live3d.py --replay runs/live-iris/stream.jsonl   # freeze a finished one

The file is the interface, deliberately: it means the viewer attaches to a run that is
already in progress, that a run killed half way still leaves you something to look at, and
that an offline `simulate_pass` can stream into exactly the same viewer as a live PX4 run.
"""
import argparse
import http.server
import json
import os
import pathlib
import socketserver
import sys
import threading
import time
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = r"""<!doctype html><meta charset=utf-8><title>flyby live</title>
<style>
body{background:#0d1017;color:#ddd;margin:0;font:13px ui-monospace,Menlo,monospace}
#top{display:flex;gap:14px;align-items:center;padding:6px 10px;background:#161b22;
     border-bottom:1px solid #2a3038;flex-wrap:wrap}
#top b{color:#7cc}button,select{background:#21262d;color:#ddd;border:1px solid #3a424c;
     border-radius:4px;padding:3px 8px;cursor:pointer}
#wrap{display:flex;height:calc(100vh - 38px)}
#main{flex:1;position:relative}
canvas{width:100%;height:100%;display:block}
#hud{position:absolute;top:8px;left:8px;background:rgba(13,16,23,.82);padding:8px 10px;
     border:1px solid #2a3038;border-radius:5px;font-size:12px;line-height:1.45}
#hud td{padding:0 8px 0 0;color:#9aa4b2}#hud td.v{color:#eee}
#side{width:250px;background:#161b22;border-left:1px solid #2a3038;padding:8px;overflow:auto}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#a00;margin-right:6px}
.dot.ok{background:#0a0}
</style>
<div id=top>
  <b>flyby live 3D</b>
  <button id=play>&#9654; play</button>
  <input id=spd type=range min=0.1 max=8 step=0.1 value=1 style=width:80px">
  <span id=spdv>1x</span>
  <button id=follow>follow: on</button>
  <button id=topdown>top-down</button>
  <label>vert exag <input id=vex type=range min=0.1 max=8 step=0.1 value=1 style=width:70px"></label>
  <span id=vexv>1.0</span>
  <span id=status class=dot></span><span id=statusTxt>waiting for telemetry</span>
</div>
<div id=wrap>
  <div id=main><canvas id=cv></canvas><div id=hud></div></div>
  <div id=side><b>point</b><div id=spec style=margin:4px 0 12px;color:#9aa4b2></div>
  <b>score</b><div id=score></div></div>
</div>
<script>
const cv=document.getElementById('cv'), cx2=cv.getContext('2d');
let P=[], SEG=[], SPEC=null, playing=true, speed=1, follow=true, topdown=false;
let t=0, dragging=false, px=0, py=0;
// Default vertical exaggeration is deliberately large. Every ground approach here is a
// descent of ~100 m over 600-3000 m of range, so at 1:1 the entire vertical story -- the
// glide slope, the arrival angle, whether it hit the ground early or late -- is a tenth of a
// pixel. The default is chosen so that view is legible without touching the slider.
vex=8;
// Camera state as its own object. It used to be smuggled through a function named `cam`,
// which is both a function and a property of window -- and referencing it as a bare object
// after the function was renamed threw `cam is not defined` in the render loop, which took
// the whole page down with a blank canvas.
const CAM = {az:-0.9, el:0.45, dist:2.2};

// world: x east, y north, z UP (metres). Same as viz.py's player.
function buildCam(){
  if(!P.length) return null;
  let mnx=1e9,mxx=-1e9,mny=1e9,mxy=-1e9,mnz=1e9,mxz=-1e9;
  for(const p of live()){ if(!p||p.length<5) continue;
                    mnx=Math.min(mnx,p[0]);mxx=Math.max(mxx,p[0]);
                    mny=Math.min(mny,p[1]);mxy=Math.max(mxy,p[1]);
                    mnz=Math.min(mnz,p[2]);mxz=Math.max(mxz,p[2]);}
  const cx=(mnx+mxx)/2, cy=(mny+mxy)/2;
  const span=Math.max(mxx-mnx,mxy-mny,60)*1.6;
  const az=CAM.az, el=CAM.el, dist=CAM.dist;
  const cx0=cx, cy0=cy, cz0=(mnz+mxz)/2;
  if(topdown){az=0; el=1.5533; dist=Math.min(dist,1.5);}
  if(follow && P.length){
    const cur=live()[nearest()];
    // pan the camera with the aircraft so it stays in frame
    window.__followX=(cur[0]-cx); window.__followY=(cur[1]-cy);
  }
  const fx=window.__followX||0, fy=window.__followY||0;
  const ce=Math.cos(el), se=Math.sin(el), ca=Math.cos(az), sa=Math.sin(az);
  const cz=cz0*vex;
  const eye=[cx0+fx + dist*span*ce*sa, cy0+fy + dist*span*ce*ca, cz + dist*span*se];
  let f=[cx0+fx-eye[0], cy0+fy-eye[1], cz-eye[2]];
  let fl=Math.hypot(f[0],f[1],f[2]); f=f.map(v=>v/fl);
  let r=[f[1],-f[0],0]; let rl=Math.hypot(r[0],r[1],r[2])||1; r=r.map(v=>v/rl);
  const u=[r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
  return {eye,f,r,u,mnx,mxx,mny,mxy,cz};
}
// The point this sample was aiming at, as [east, north, up] plus the pass bearing. Taken
// from the row itself (position minus its along/cross offset), not from SPEC: SPEC is the
// meta file, which describes only the LATEST case, and every other case drawn against it
// appears to miss a point it was never flying at.
function target(p){
  if(p && p[7]!=null && p[8]!=null)
    return {e:p[0]-p[8], n:p[1]-p[7],
            u:(p[12]!=null? p[12] : (SPEC&&SPEC.ned? -SPEC.ned[2] : 0)),
            brg:(p[13]!=null? p[13] : (SPEC? SPEC.angle : 0))};
  if(SPEC && SPEC.ned) return {e:SPEC.ned[1], n:SPEC.ned[0], u:-SPEC.ned[2], brg:SPEC.angle};
  return null;
}
// Which case is on screen, held by IDENTITY (its first sample object), not by index:
// resegment() rebuilds SEG on every message and trims old cases, so an index silently
// points at a different case a moment later. SHOW=null means "pick the newest".
// Playback finishes the case on screen -- through the gate, plus a short look at it --
// before moving on, and then jumps to the NEWEST case rather than queueing every one: an
// unpaced benchmark makes cases far faster than anyone can watch, and an approach replaced
// before it arrives looks exactly like one that never arrives. PIN (set by '[' / ']')
// stops the auto-advance so a case can be inspected; ']' past the newest releases it.
let SHOW=null, PIN=false, ENDED=0;
function segIdx(){
  if(!SEG.length) return -1;
  if(SHOW) for(let i=SEG.length-1;i>=0;i--) if(SEG[i][0]===SHOW) return i;
  SHOW=SEG[SEG.length-1][0]; t=0; ENDED=0;
  return SEG.length-1;
}
function live(){ const i=segIdx(); return i<0? P : SEG[i]; }
function showSeg(i){ SHOW=SEG[i][0]; t=0; ENDED=0; }
function stepSeg(d){ if(!SEG.length) return;
  const cur=segIdx(), nx=cur+d;
  if(nx>=SEG.length){ PIN=false; showSeg(SEG.length-1); }
  else { PIN=true; showSeg(Math.max(0,nx)); }
  statusLine(); }
// When the case on screen is "done": a few seconds past its gate (the rest is the fly-on
// tail), or its last sample if it has no gate.
function endT(L){
  for(const p of L) if(p[11]) return Math.min(L[L.length-1][4], p[4]+3);
  return L[L.length-1][4];
}
function nearest(){
  const L=live(); if(!L.length) return 0;
  let best=0,bd=1e18;
  for(let i=0;i<L.length;i++){ if(!L[i]||L[i].length<5) continue;
                               const d=Math.abs(L[i][4]-t); if(d<bd){bd=d;best=i;}}
  return best;
}
function monotonicTimes(){
  // A stream written by an older build (or across a phase boundary) can still contain a
  // time that goes backwards. Nothing downstream cares -- the trail is drawn in array
  // order, and the playhead is clamped to the last sample -- but it is worth saying so.
  for(let i=1;i<P.length;i++) if(P[i]&&P[i-1]&&P[i][4]<P[i-1][4]) return false;
  return true;
}
function proj(p,C,w,h){
  const q=[p[0]-C.eye[0],p[1]-C.eye[1],p[2]*vex-C.eye[2]];
  const z=q[0]*C.f[0]+q[1]*C.f[1]+q[2]*C.f[2];
  if(z<=1) return null;
  const x=q[0]*C.r[0]+q[1]*C.r[1]+q[2]*C.r[2];
  const y=q[0]*C.u[0]+q[1]*C.u[1]+q[2]*C.u[2];
  const F=Math.min(w,h)*0.95;
  return [w/2+x*F/z, h/2-y*F/z, z];
}
function draw(){
  cv.width=cv.clientWidth*devicePixelRatio; cv.height=cv.clientHeight*devicePixelRatio;
  const w=cv.width,h=cv.height;
  const g=cx2.createLinearGradient(0,0,0,h);
  g.addColorStop(0,'#0d1017'); g.addColorStop(1,'#181d26');
  cx2.fillStyle=g; cx2.fillRect(0,0,w,h);
  const C=buildCam(); if(!C) return;
  // ground grid at z=0
  cx2.strokeStyle='#20262f'; cx2.lineWidth=1; cx2.beginPath();
  for(let x=Math.floor((C.mnx-300)/200)*200;x<=C.mxx+300;x+=200){
    const a=proj([x,C.mny-300,0],C,w,h), b=proj([x,C.mxy+300,0],C,w,h);
    if(a&&b){cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);}
  }
  for(let y=Math.floor((C.mny-300)/200)*200;y<=C.mxy+300;y+=200){
    const a=proj([C.mnx-300,y,0],C,w,h), b=proj([C.mxx+300,y,0],C,w,h);
    if(a&&b){cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);}
  }
  cx2.stroke();
  // the pass line and the point, once we know them
  const T=target(live()[nearest()]);
  if(T){
    // World here is [east, north, up]. The bearing is from north, so its east component is
    // sin and its north component cos. Feeding SPEC.ned ([north, east, down]) straight in
    // used to draw the point mirrored across the diagonal, i.e. somewhere it never was.
    const d=T.brg*Math.PI/180;
    const D=[Math.sin(d),Math.cos(d)];
    const L=Math.max(C.mxx-C.mnx,C.mxy-C.mny,60)*1.2;
    const a=proj([T.e-D[0]*L,T.n-D[1]*L,T.u],C,w,h);
    const b=proj([T.e+D[0]*L,T.n+D[1]*L,T.u],C,w,h);
    if(a&&b){cx2.strokeStyle='#3d5a80';cx2.setLineDash([6,5]);cx2.beginPath();
             cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);cx2.stroke();cx2.setLineDash([]);}
    const g0=proj([T.e,T.n,T.u],C,w,h);
    if(g0){cx2.fillStyle='#e0a030';cx2.beginPath();cx2.arc(g0[0],g0[1],6,0,7);cx2.fill();
            cx2.fillStyle='#fff';cx2.font='11px monospace';
            cx2.fillText('P',g0[0]+8,g0[1]-6);}
  }
  // trail, coloured by time
  cx2.lineWidth=2;
  // Each time segment is drawn as its own polyline, and only within a segment, so no
  // renderer state can join the end of one phase to the start of another. Drawing across a
  // backwards time jump invents a teleport the aircraft never made, and a rendering bug must
  // never look like a flight bug.
  // Only the case on screen plus a few recent ones as faint context. Drawing all 240 kept
  // cases is ~300k strokes a frame and freezes the page; and the bright one has to be the
  // case being PLAYED, which is no longer always the newest.
  const si=segIdx();
  const segs = SEG.length? SEG.slice(-8).filter(L=>L!==SEG[si]).concat([SEG[si]]) : [P];
  for(let k=0;k<segs.length;k++){
    const L=segs[k], isLive=(k===segs.length-1);
    for(let i=1;i<L.length;i++){
      if(!L[i-1]||!L[i]||L[i-1].length<5||L[i].length<5) continue;
      const gap=Math.abs(L[i][4]-L[i-1][4]);
      if(gap>2.0||gap<=0) continue;
      const a=proj(L[i-1],C,w,h), b=proj(L[i],C,w,h);
      if(!a||!b) continue;
      const f=L[i][4]/Math.max(1, L[L.length-1][4]||1);
      cx2.strokeStyle=isLive?`hsl(${(1-f)*190},70%,60%)`:'#3a4a5a';
      cx2.beginPath(); cx2.moveTo(a[0],a[1]); cx2.lineTo(b[0],b[1]); cx2.stroke();
    }
  }
  // The gate: where the pass was scored. The trail after it is the law flying ON past the
  // point, which is correct behaviour and looks exactly like a miss if not marked.
  const L0=live();
  let gi=-1;
  for(let i=0;i<L0.length;i++){ if(L0[i] && L0[i].length>11 && L0[i][11]){ gi=i; break; } }
  if(gi>=0 && L0[gi]){
    const gs=proj(L0[gi],C,w,h);
    if(gs){
      cx2.strokeStyle='#e0a030'; cx2.lineWidth=2; cx2.beginPath();
      cx2.moveTo(gs[0]-11,gs[1]); cx2.lineTo(gs[0]+11,gs[1]);
      cx2.moveTo(gs[0],gs[1]-11); cx2.lineTo(gs[0],gs[1]+11); cx2.stroke();
      cx2.fillStyle='#e0a030'; cx2.font='12px monospace';
      cx2.fillText('GATE (scored here)', gs[0]+14, gs[1]+4);
    }
  }
  const i=nearest(), p=live()[i];
  if(p){
    const s=proj(p,C,w,h);
    if(s){cx2.fillStyle='#fff';cx2.beginPath();cx2.arc(s[0],s[1],5,0,7);cx2.fill();
          cx2.strokeStyle='#4af';cx2.lineWidth=2;cx2.beginPath();
          cx2.arc(s[0],s[1],9,0,7);cx2.stroke();}
    hud(p);
  }
}
// A live viewer must never die on one bad sample. An exception thrown out of draw() stops
// requestAnimationFrame for good, so the canvas freezes and the whole page looks dead -- a
// single undefined field costs every subsequent frame. Everything here is therefore total:
// missing numbers render as a dash instead of throwing.
const num=(v,d=0)=>(v===undefined||v===null||Number.isNaN(v))?null:(+v);
const fmt=(v,d=1)=>{const x=num(v); return x===null?'-':x.toFixed(d);};
function hud(p){
  // The stream's `along`/`cross` are RAW north/east differences from the point, not
  // pass-frame components. Labelling them "along-track error" and "cross-track error" is a
  // lie that only shows up on a badly-tracking pass: a case that missed by 5 m of true
  // cross-track reported 1295 m, because by the end of the fly-on tail it really was
  // 1295 m east of the point. The pass bearing is known, so project properly.
  let al=null, cr=null, tgt=null;
  const T=target(p);
  if(T){
    const a=T.brg*Math.PI/180, d=[Math.cos(a),Math.sin(a)], r=[-Math.sin(a),Math.cos(a)];
    const dn=p[1]-T.n, de=p[0]-T.e;
    al=dn*d[0]+de*d[1]; cr=dn*r[0]+de*r[1]; tgt=Math.hypot(dn,de);
  }
  const rows=[['t',fmt(p[4],1)+' s'],['altitude',fmt(p[2],1)+' m'],
    ['ground speed',fmt(p[5],2)+' m/s'],['climb rate',(-(num(p[6])||0)).toFixed(2)+' m/s']];
  if(al!==null) rows.push(['distance to point',fmt(tgt,2)+' m']);
  if(cr!==null){
    rows.push(['along-track err',fmt(al,2)+' m']);
    rows.push(['cross-track err',fmt(cr,2)+' m']);
  }
  if(p[9]!=null) rows.push(['commanded vz',p[9].toFixed(2)+' m/s']);
  if(p[10]!=null) rows.push(['feasible',p[10]?'yes':'<b style=color:#f66>NO</b>']);
  document.getElementById('hud').innerHTML=
    rows.map(r=>`<table><tr><td>${r[0]}</td><td class=v>${r[1]}</td></tr></table>`).join('');
}
function setSpec(s){
  if(!s) return;
  SPEC=s;
  document.getElementById('spec').innerHTML=
    ['track '+Number(s.angle).toFixed(1)+'°',
     'arrival speed '+Number(s.speed).toFixed(2)+' m/s',
     (s.fp!=null?'demanded fp '+Number(s.fp).toFixed(1)+'°':''),
     (s.motion||'')].filter(Boolean).join('<br>');
}

// PUSH, not poll. One EventSource, appended to as samples arrive -- no re-reading the run,
// no re-sending it, no 200 ms of latency. `/data` is still there for the initial snapshot,
// because an SSE stream carries only what happens AFTER it connects.
function resegment(){
  SEG=[]; let cur=[];
  for(const p of P){ if(cur.length && p[4] < cur[cur.length-1][4]){SEG.push(cur); cur=[];}
                      cur.push(p); }
  if(cur.length) SEG.push(cur);
  // Keep a bounded history: a long benchmark would otherwise grow SEG without limit and
  // take the whole page down. 240 cases is far more than anyone scrolls through.
  // P is trimmed to match, or it grows without bound and this rebuild gets slower with
  // every message until the page stalls.
  // The case on screen is kept even when it is the oldest: an unpaced run outdates 240
  // cases in seconds, and trimming it away mid-approach is the same strobe again.
  if(SEG.length>240){
    let k=-1; if(SHOW) for(let i=0;i<SEG.length;i++) if(SEG[i][0]===SHOW){ k=i; break; }
    const keep=SEG.slice(-239);
    SEG=(k>=0 && k<SEG.length-239)? [SEG[k]].concat(keep) : SEG.slice(-240);
    P=SEG.flat();
  }
  // The playhead is NOT touched here. Resetting it on every arriving case meant an unpaced
  // benchmark only ever showed the first second of each approach. loop() decides when to
  // move on.
}
function statusLine(){
  const L=live();
  document.getElementById('status').className='dot ok';
  document.getElementById('statusTxt').textContent=
    P.length+' samples, t='+(L.length?L[L.length-1][4].toFixed(1):'0.0')+' s'
    +(SEG.length>1 ? '   case '+(segIdx()+1)+'/'+SEG.length
                   +(PIN? ' (pinned, ] past newest to follow)' : ' (following)')
                   +'   [ / ] to step' : '')
    + (LIVE? '   live':'');
}
addEventListener('keydown',e=>{ if(e.key==='[') stepSeg(-1); else if(e.key===']') stepSeg(1); });
async function snapshot(){
  try{
    const j=await (await fetch('/data?t='+Date.now(),{cache:'no-store'})).json();
    if(j.spec && !SPEC) setSpec(j.spec);
    // Accept any change, not just a longer one. Every case truncates the stream file, so the
    // next case is usually SHORTER than the previous one -- the length test silently threw
    // those away and the page sat frozen on a stale trajectory, which looks exactly like a
    // broken stream.
    if(j.samples && j.samples.length && (j.samples.length!==P.length || j.samples[0]!==P[0])){
      P=j.samples; SHOW=null; PIN=false; resegment(); statusLine(); }
  }catch(e){}
}
let LIVE=false;
function connect(){
  const es=new EventSource('/events');
  es.onopen=()=>{ LIVE=true; statusLine(); };
  es.onmessage=ev=>{
    let j; try{ j=JSON.parse(ev.data); }catch(e){ return; }
    if(j.spec && !SPEC) setSpec(j.spec);
    if(j.samples && j.samples.length){
      // Append, never replace. This is the whole reason to use a push stream.
      for(const r of j.samples) P.push(r);
      resegment(); statusLine();
    }
  };
  es.onerror=()=>{ LIVE=false;
    document.getElementById('status').className='dot';
    document.getElementById('statusTxt').textContent='reconnecting...'; };
}
snapshot().then(connect);
document.getElementById('play').onclick=()=>{playing=!playing;
  document.getElementById('play').textContent=playing?'❚❚ pause':'▶ play';};
document.getElementById('spd').oninput=e=>{speed=+e.target.value;
  document.getElementById('spdv').textContent=speed.toFixed(1)+'x';};
document.getElementById('follow').onclick=()=>{follow=!follow;
  document.getElementById('follow').textContent='follow: '+(follow?'on':'off');};
document.getElementById('topdown').onclick=()=>{topdown=!topdown;
  document.getElementById('topdown').textContent=topdown?'oblique':'top-down';};
document.getElementById('vex').oninput=e=>{vex=+e.target.value;
  document.getElementById('vexv').textContent=vex.toFixed(1);};
cv.addEventListener('wheel',e=>{e.preventDefault();
  CAM.dist=Math.max(0.25,Math.min(8, CAM.dist*(1+e.deltaY*0.001)));}, {passive:false});
cv.addEventListener('mousedown',e=>{dragging=true;px=e.clientX;py=e.clientY;});
addEventListener('mouseup',()=>{dragging=false;});
addEventListener('mousemove',e=>{ if(!dragging)return;
  CAM.az+=(e.clientX-px)*0.006;
  CAM.el=Math.max(0.03,Math.min(1.5, CAM.el+(e.clientY-py)*0.006));
  px=e.clientX; py=e.clientY; });
function loop(){
  const L=live();
  if(playing && L.length){
    const last=L[L.length-1][4], i=segIdx();
    t += 0.016*speed*4;
    if(t > last+0.01) t = last;          // live: stay glued to the newest sample
    // Done with this case and a newer one exists: hold on the arrival for a moment, then
    // jump to the newest. A single growing case (a live flight) never advances.
    if(!PIN && i < SEG.length-1 && t >= endT(L)){
      if(!ENDED) ENDED=performance.now();
      else if(performance.now()-ENDED > 800){ showSeg(SEG.length-1); statusLine(); }
    }
  }
  draw(); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the page, the telemetry file, and a live push stream."""

    # Both halves of the connection share one stream reader, or two of them interleave on
    # the same file and each misses the other's samples.
    READER = None

    def __init__(self, *a, stream_path=None, meta_path=None, reader=None, **kw):
        self.stream_path = pathlib.Path(stream_path) if stream_path else None
        self.meta_path = pathlib.Path(meta_path) if meta_path else None
        if reader is not None:
            Handler.READER = reader
        super().__init__(*a, directory=None, **kw)

    def log_message(self, *a):        # keep the flight output clean
        pass

    def do_GET(self):                                       # noqa: N802
        if self.path.startswith("/events"):
            self.serve_events()
            return
        if self.path.startswith("/data"):
            self.send_json(self.read_telemetry())
            return
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # No-store on the PAGE as well as the data. The page is the thing that gets
            # edited while a flight is in progress, and a browser holding a cached copy of
            # the old one shows a blank canvas against perfectly good telemetry -- with no
            # error anywhere, because the stale page is internally consistent and the data
            # it is reading uses keys it has never heard of.
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def serve_events(self):
        """Push new samples as they land, over Server-Sent Events.

        SSE rather than a WebSocket because this is one-directional: telemetry only ever
        flows server -> browser, and SSE is plain HTTP with no framing protocol, no
        handshake, and automatic reconnection in the browser. `http.server` can hold a
        chunked response open and write into it, which is all this needs. A WebSocket would
        work too but means implementing RFC 6455 framing by hand for no gain here.

        The other half of the point is that this sends only what is NEW. Polling re-read and
        re-sent the whole run five times a second -- quadratic in the length of the flight,
        which at a 150 s pass is several hundred megabytes of HTTP to draw a line.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last = Handler.READER.pos if Handler.READER is not None else 0
        sent_spec = object()
        try:
            self.wfile.write(b": open\n\n")
            self.wfile.flush()
            while True:
                spec = _read_meta(self.meta_path)
                rows, last = _read_new(self.stream_path, last)
                # Send on NEW SAMPLES, or when the spec actually changed. The guard used to
                # be `if rows or spec is not None`, and spec is essentially always non-None,
                # so it never throttled: the handler spun as fast as it could, emitting SSE
                # frames with empty `samples` lists thousands of times a second. An
                # EventSource cannot keep up with that and the page stops painting, which
                # reads as "nothing moves in the UI". The else-branch sleep is what makes
                # this a poll loop rather than a spin loop.
                changed = rows or (spec is not None and spec != sent_spec)
                if changed:
                    sent_spec = spec
                    # Whole object, NOT braces stripped. The `[1:-1]` that was here made
                    # every frame unparseable JSON, so the client silently discarded all of
                    # them and the page just sat there -- which looks identical to "the push
                    # stream does not work". An SSE data field only has to avoid newlines,
                    # and json.dumps never emits one.
                    payload = json.dumps({"samples": rows, "spec": spec})
                    self.wfile.write(b"data: " + payload.encode() + b"\n\n")
                    self.wfile.flush()
                else:
                    time.sleep(0.02)
                    # A comment frame keeps the connection warm through proxies that would
                    # otherwise drop an idle stream.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_telemetry(self):
        """The stream file, as the page wants it.

        Read whole each poll, not incrementally: the file is a few hundred kB at most for
        a 150 s run, and a partial read that skips a line mid-object is a far worse failure
        mode than re-reading a small file. Truncated trailing lines are dropped, because a
        reader attached mid-write sees exactly that.
        """
        # Both `spec` and `meta` live in the .meta.json sidecar. This used to pair the
        # sidecar with the "meta" role and the JSONL stream with the "spec" role, so
        # json.loads() was run over the JSONL (which is one object per line and therefore
        # never parses) and spec was permanently None, while `meta` got the un-unwrapped
        # {"spec":..., "meta":...} document. /data therefore reported no pass bearing, so
        # the client could not project along/cross-track and the HUD fell back to raw
        # north/east deltas.
        spec, meta = None, None
        if self.meta_path is not None and pathlib.Path(self.meta_path).exists():
            try:
                doc = json.loads(pathlib.Path(self.meta_path).read_text() or "{}")
            except Exception:
                doc = {}
            if isinstance(doc, dict):
                spec = doc.get("spec")
                meta = doc.get("meta")
        out = []
        dur = 0.0
        if self.stream_path and self.stream_path.exists():
            try:
                lines = self.stream_path.read_text().splitlines()
            except Exception:
                lines = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue          # a half-written last line
                out.append([r["east"], r["north"], r["alt"], r["amsl"], r["t"],
                            r.get("gs"), r.get("vd"), r.get("along"), r.get("cross"),
                            r.get("cvz"), r.get("ok"), r.get("gate") or 0,
                            r.get("tu"), r.get("brg")])
                dur = r["t"]
        return {"samples": out, "dur": dur, "spec": spec, "meta": meta}


class StreamReader:
    """Byte offset into the stream file, so each read returns only what is new."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.pos = 0


def _read_meta(meta_path):
    if meta_path is None or not pathlib.Path(meta_path).exists():
        return None
    try:
        return json.loads(pathlib.Path(meta_path).read_text() or "{}").get("spec")
    except Exception:
        return None


def _read_new(stream_path, last):
    """(rows, new_offset). A half-written trailing line is left for next time."""
    out = []
    if stream_path is None or not stream_path.exists():
        return out, last
    try:
        with open(stream_path, "rb") as f:
            size = f.seek(0, 2)
            if size < last:            # truncated: a new flight. Start over.
                last = 0
            f.seek(last)
            blob = f.read()
    except Exception:
        return out, last
    cut = blob.rfind(b"\n")
    if cut < 0:
        return out, last
    chunk, last = blob[:cut + 1], last + cut + 1
    for raw in chunk.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            r = json.loads(raw)
        except Exception:
            continue
        # Field 11 is the gate flag. It has to be emitted here as well as in _load_state():
        # this is the SSE path, so without it the live stream could never carry a gate
        # marker even though the initial /data snapshot did.
        out.append([r.get("east"), r.get("north"), r.get("alt"), r.get("amsl"),
                    r.get("t"), r.get("gs"), r.get("vd"), r.get("along"), r.get("cross"),
                    r.get("cvz"), r.get("ok"), r.get("gate") or 0,
                    r.get("tu"), r.get("brg")])
    return out, last


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stream", nargs="?", default="runs/live/stream.jsonl")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="open a browser")
    args = ap.parse_args()
    stream = pathlib.Path(args.stream)
    meta = stream.with_suffix(".meta.json")
    reader = StreamReader(stream)

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", args.port),
                                            lambda *a, **kw: Handler(*a, stream_path=stream,
                                                                   meta_path=meta,
                                                                   reader=reader, **kw))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"live 3D viewer on {url}")
    print(f"  watching {stream}")
    print("  it works before the flight starts, and attaches to one already running.")
    print("  Ctrl-C to stop.")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url))).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())