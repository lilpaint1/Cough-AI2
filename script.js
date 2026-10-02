// ============================================================
// CoughAI — script.js
// ============================================================

const LABELS      = ['covid', 'healthy', 'symptomatic'];
const SAMPLE_RATE = 44100;
const DURATION    = 5;
const FLASK_URL   = "/predict";   // relative path → ใช้ได้ทุก host/port

// ── DOM refs ──────────────────────────────────────────────
const recordButton    = document.getElementById('record-button');
const recordText      = document.getElementById('record-text');
const uploadInput     = document.getElementById('audio-upload');
const fileNameSpan    = document.getElementById('file-name');
const recordSection   = document.getElementById('record-section');
const uploadSection   = document.getElementById('upload-section');
const inputSection    = document.getElementById('input-section');
const resultSection   = document.getElementById('result-section');
const resultLoading   = document.getElementById('result-loading');
const resultCard      = document.getElementById('result-card');
const predictionLabel = document.getElementById('prediction-label');
const confidenceScore = document.getElementById('confidence-score');
const statusMessage   = document.getElementById('status-message');
const probContainer   = document.getElementById('probability-bar-container');
const resetButton     = document.getElementById('reset-button');
const audioBar        = document.getElementById('audio-bar');
const waveformLabel   = document.getElementById('waveform-label');
const messageModal    = document.getElementById('message-modal');
const modalTitle      = document.getElementById('modal-title');
const modalMessage    = document.getElementById('modal-message');
const modalOkButton   = document.getElementById('modal-ok-button');
const resultBadge     = document.getElementById('result-badge');
const badgeIcon       = document.getElementById('badge-icon');
const dropZone        = document.getElementById('drop-zone');

// ── new refs ──────────────────────────────────────────────
const recoText        = document.getElementById('reco-text');
const hospitalBtn     = document.getElementById('hospital-btn');
const callBtn         = document.getElementById('call-btn');
const callModal       = document.getElementById('call-modal');
const callCancel      = document.getElementById('call-cancel');
const customTel       = document.getElementById('custom-tel');
const customCallBtn   = document.getElementById('custom-call-btn');

let mediaRecorder, audioChunks = [], audioContext, analyserNode, animationFrameId;

// ── i18n helpers ──────────────────────────────────────────
const T = (k) => (window.t ? window.t(k) : k);
let lastResult = null;   // เก็บผลล่าสุดไว้ render ใหม่ตอนสลับภาษา

function renderDate() {
    const el = document.getElementById('app-date');
    if (!el) return;
    const lang = window.CoughLang ? window.CoughLang.get() : 'th';
    const loc  = lang === 'en' ? 'en-US' : 'th-TH';
    el.textContent = new Date().toLocaleDateString(loc, { weekday:'long', day:'numeric', month:'long' });
}

// render ส่วน dynamic ใหม่เมื่อผู้ใช้สลับภาษา
document.addEventListener('coughai:langchange', () => {
    renderDate();
    if (lastResult && resultCard.style.display !== 'none') displayResult(lastResult);
});

// ── Splash → App ──────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    renderDate();

    setTimeout(() => {
        const splash = document.getElementById('splash');
        const app    = document.getElementById('app');
        splash.classList.add('splash-out');
        setTimeout(() => {
            splash.style.display = 'none';
            app.classList.add('app-visible');
        }, 500);
    }, 2200);
});

// ── Modal ─────────────────────────────────────────────────
function showModal(title, message) {
    modalTitle.textContent   = title;
    modalMessage.textContent = message;
    messageModal.style.display = 'flex';
}
modalOkButton.addEventListener('click', () => messageModal.style.display = 'none');

// ── Record ────────────────────────────────────────────────
recordButton.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
    else startRecording();
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder  = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks    = [];

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            cancelAnimationFrame(animationFrameId);
            const blob    = new Blob(audioChunks, { type: 'audio/webm' });
            const wavBlob = await convertToWav(blob);
            processAudio(wavBlob);
            stream.getTracks().forEach(t => t.stop());
        };

        mediaRecorder.start();
        recordText.textContent = T('app.recording');
        recordButton.classList.add('recording');
        if (waveformLabel) waveformLabel.textContent = T('app.wave.recording');
        uploadInput.disabled = true;
        visualizeAudio();

        setTimeout(() => { if (mediaRecorder.state === 'recording') mediaRecorder.stop(); }, DURATION * 1000);
    } catch (err) {
        showModal(T('app.error.title'), `${T('app.error.mic')}${err.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
    recordText.textContent = T('app.record.start');
    recordButton.classList.remove('recording');
    if (waveformLabel) waveformLabel.textContent = T('app.wave.idle');
    audioBar.style.width = '0%';
    uploadInput.disabled = false;
}

function visualizeAudio() {
    const data = new Uint8Array(analyserNode.frequencyBinCount);
    const tick = () => {
        analyserNode.getByteFrequencyData(data);
        const avg = data.reduce((a,b)=>a+b,0) / data.length;
        audioBar.style.width = `${(avg/255)*100}%`;
        animationFrameId = requestAnimationFrame(tick);
    };
    tick();
}

// ── Upload ────────────────────────────────────────────────
uploadInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    fileNameSpan.textContent = file.name;
    convertFileToWav(file).then(wav => processAudio(wav));
});

if (dropZone) {
    dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file?.type.startsWith('audio/')) {
            fileNameSpan.textContent = file.name;
            convertFileToWav(file).then(wav => processAudio(wav));
        } else {
            showModal(T('app.error.filetype.title'), T('app.error.filetype.msg'));
        }
    });
}

// ── Audio conversion ──────────────────────────────────────
async function convertToWav(blob) {
    const ab  = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(ab);
    return new Blob([audioBufferToWav(buf, 1, SAMPLE_RATE)], { type: 'audio/wav' });
}
async function convertFileToWav(file) {
    return file.type === 'audio/wav' ? file : convertToWav(file);
}
function audioBufferToWav(buffer, numChannels, sampleRate) {
    const ch  = [];
    for (let i=0;i<numChannels;i++) ch.push(buffer.getChannelData(i % buffer.numberOfChannels));
    const il  = interleave(ch);
    const wav = new ArrayBuffer(44 + il.length*2);
    const v   = new DataView(wav);
    writeString(v,0,'RIFF'); v.setUint32(4,36+il.length*2,true); writeString(v,8,'WAVE');
    writeString(v,12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true);
    v.setUint16(22,numChannels,true); v.setUint32(24,sampleRate,true);
    v.setUint32(28,sampleRate*numChannels*2,true); v.setUint16(32,numChannels*2,true);
    v.setUint16(34,16,true); writeString(v,36,'data'); v.setUint32(40,il.length*2,true);
    let o=44;
    for (let i=0;i<il.length;i++,o+=2){const s=Math.max(-1,Math.min(1,il[i]));v.setInt16(o,s<0?s*0x8000:s*0x7FFF,true);}
    return wav;
}
function interleave(ch) {
    const r=new Float32Array(ch[0].length*ch.length);
    for (let i=0;i<ch[0].length;i++) for(let j=0;j<ch.length;j++) r[i*ch.length+j]=ch[j][i];
    return r;
}
function writeString(v,o,s){for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));}

// ── Process → API ─────────────────────────────────────────
async function processAudio(audioBlob) {
    inputSection.style.display = 'none';
    resultSection.style.display = 'block';
    resultLoading.style.display = 'flex';
    resultCard.style.display    = 'none';
    statusMessage.textContent   = T('app.status.analyzing');

    try {
        const fd = new FormData();
        fd.append('file', audioBlob, 'cough.wav');
        const res = await fetch(FLASK_URL, { method:'POST', body:fd });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        await new Promise(r => setTimeout(r, 350));
        displayResult(data);
    } catch (err) {
        console.error(err);
        displayError();
    }
}

// ── Display result ────────────────────────────────────────
function displayResult(data) {
    resultLoading.style.display = 'none';
    resultCard.style.display    = 'flex';
    lastResult = data;

    const cls   = data.classification.toLowerCase();
    const probs = data.probabilities;
    const top   = probs.reduce((m,p) => Math.max(m, p.score), 0);

    const map = {
        healthy:     { text:'Healthy',     color:'var(--teal)',   bg:'var(--teal-dim)',   icon:'<polyline points="20 6 9 17 4 12"/>' },
        covid:       { text:'COVID-19',    color:'var(--red)',    bg:'var(--red-dim)',    icon:'<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>' },
        symptomatic: { text:'Symptomatic', color:'var(--yellow)', bg:'var(--yellow-dim)', icon:'<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>' },
    };
    const m = map[cls] || { text: cls.toUpperCase(), color:'var(--text2)', bg:'rgba(255,255,255,0.08)', icon:'<circle cx="12" cy="12" r="10"/>' };

    predictionLabel.textContent  = map[cls] ? T('label.' + cls) : m.text;
    predictionLabel.style.color  = m.color;
    resultBadge.style.background = m.bg;
    badgeIcon.innerHTML          = m.icon;
    badgeIcon.style.stroke       = m.color;
    confidenceScore.textContent  = `${T('app.confidence')} ${(top*100).toFixed(1)}%`;

    // probability bars
    probContainer.innerHTML = '';
    const colorMap = { healthy:'healthy', covid:'covid', symptomatic:'symptomatic' };
    probs.forEach(p => {
        const row  = document.createElement('div'); row.className = 'prob-row';
        const name = document.createElement('span'); name.className = 'prob-name';
        const pk = p.label.toLowerCase();
        name.textContent = ['healthy','covid','symptomatic'].includes(pk) ? T('label.' + pk) : p.label;
        const trk  = document.createElement('div'); trk.className = 'prob-track';
        const fill = document.createElement('div');
        const cls2 = colorMap[p.label.toLowerCase()] || 'default';
        fill.className = `prob-fill ${cls2}`;
        fill.style.width = '0%';
        trk.appendChild(fill);
        const pct = document.createElement('span'); pct.className = 'prob-pct'; pct.textContent = `${(p.score*100).toFixed(1)}%`;
        row.append(name, trk, pct);
        probContainer.appendChild(row);
        requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${p.score*100}%`; }));
    });

    // ── คำแนะนำ ──
    // backend ส่ง recommendation มาเป็นภาษาไทย → ใช้เมื่ออยู่โหมดไทย
    // โหมดอังกฤษหรือไม่มีข้อมูล → ใช้คำแนะนำจาก dictionary
    if (recoText) {
        const lang = window.CoughLang ? window.CoughLang.get() : 'th';
        const fallback = ['healthy','covid','symptomatic'].includes(cls) ? T('reco.' + cls) : '—';
        recoText.textContent = (lang === 'th' && data.recommendation) ? data.recommendation : fallback;
    }
}

function displayError() {
    resultLoading.style.display = 'none';
    resultCard.style.display    = 'flex';
    lastResult = null;
    predictionLabel.textContent  = T('app.fail.label');
    predictionLabel.style.color  = 'var(--text2)';
    resultBadge.style.background = 'rgba(255,255,255,0.06)';
    badgeIcon.innerHTML          = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>';
    badgeIcon.style.stroke       = 'var(--text2)';
    confidenceScore.textContent  = T('app.fail.retry');
    probContainer.innerHTML      = '';
    if (recoText) recoText.textContent = T('app.fail.reco');
}

// ── รพ. ใกล้ฉัน → เปิด Google Maps ทันที (ไม่ต้องรอ geolocation) ───────────
// เปิด URL จริงเลยตอนกด → ไม่มีแท็บเปล่า about:blank อีก
// Google Maps คำว่า "ใกล้ฉัน" จะใช้ตำแหน่งอุปกรณ์ผ่าน Google เอง
hospitalBtn?.addEventListener('click', () => {
    const url = `https://www.google.com/maps/search/${encodeURIComponent(T('maps.hospitalNear'))}`;
    const win = window.open(url, '_blank', 'noopener');
    if (!win) window.location.href = url;   // เผื่อ popup ถูกบล็อก → เปิดในแท็บเดิม
});

// ── โทรปรึกษาแพทย์ (call picker sheet) ────────────────────
callBtn?.addEventListener('click', () => { callModal.style.display = 'flex'; });
callCancel?.addEventListener('click', () => { callModal.style.display = 'none'; });
callModal?.addEventListener('click', e => { if (e.target === callModal) callModal.style.display = 'none'; });
customCallBtn?.addEventListener('click', () => {
    const num = (customTel.value || '').replace(/[^0-9+]/g, '');
    if (!num) { showModal(T('app.error.nonum.title'), T('app.error.nonum.msg')); return; }
    window.location.href = `tel:${num}`;
});

// ── Reset ─────────────────────────────────────────────────
resetButton.addEventListener('click', () => {
    resultSection.style.display = 'none';
    inputSection.style.display  = '';
    resultCard.style.display    = 'none';
    fileNameSpan.textContent    = '';
    audioBar.style.width        = '0%';
    if (waveformLabel) waveformLabel.textContent = T('app.wave.idle');
    uploadInput.value           = '';
    recordButton.disabled       = false;
    recordButton.classList.remove('recording');
    recordText.textContent      = T('app.record.start');
});
