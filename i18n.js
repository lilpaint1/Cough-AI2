// ============================================================
// CoughAI — i18n.js  (ระบบสลับภาษา TH / EN ใช้ร่วมกันทุกหน้า)
//   • default = ไทย (จำค่าที่ผู้ใช้เลือกใน localStorage)
//   • HTML: ใส่ data-i18n="key" / data-i18n-html="key" / data-i18n-ph="key"
//   • JS:   เรียก window.t('key') และฟัง event 'coughai:langchange'
// ============================================================
(function () {
    const STORE_KEY = 'coughai-lang';
    const DEFAULT   = 'th';

    const DICT = {
        th: {
            // ── nav ──
            'nav.home': 'หน้าแรก',
            'nav.app': 'แอป',
            'nav.dashboard': 'แดชบอร์ด',

            // ── labels / risk (ใช้ร่วม app + dashboard) ──
            'label.healthy': 'สุขภาพดี',
            'label.covid': 'โควิด-19',
            'label.symptomatic': 'มีอาการ',
            'risk.HIGH': 'เสี่ยงสูง',
            'risk.MEDIUM': 'เฝ้าระวัง',
            'risk.LOW': 'ปกติ',

            // ════ HOMEPAGE ════
            'doc.home.title': 'CoughAI — คัดกรองสุขภาพทางเดินหายใจ',
            'home.cta': 'เริ่มวิเคราะห์ →',
            'home.badge': 'ขับเคลื่อนด้วย AI · วิเคราะห์เรียลไทม์',
            'home.headline': 'เปลี่ยนเสียงไอของคุณ<br>ให้เป็น <em>สัญญาณสุขภาพล่วงหน้า</em>',
            'home.sub': 'AI วิเคราะห์เสียงไอของคุณในไม่กี่วินาที<br class="desktop-br">เพื่อประเมินความเสี่ยงทางเดินหายใจ — ไม่ต้องไปคลินิก',
            'home.start': 'เริ่มวิเคราะห์',
            'home.how': 'วิธีใช้งาน',
            'home.trust.privacy': 'ความเป็นส่วนตัวมาก่อน',
            'home.trust.fast': 'ผลใน <5 วิ',
            'home.trust.noupload': 'ไม่ต้องอัปโหลด',
            'home.visual.analyzing': 'กำลังวิเคราะห์รูปแบบเสียง',
            'home.visual.diagnosis': 'ผลการคัดกรอง',
            'home.chip.sound': 'ตรวจพบเสียง',
            'home.chip.model': 'โมเดล AI พร้อม',
            'home.how.eyebrow': 'ขั้นตอนง่าย ๆ',
            'home.how.title': '3 ขั้นตอนสู่ผลวิเคราะห์',
            'home.how.sub': 'ไม่ต้องไปหาหมอ ไม่ต้องตรวจแล็บ แค่เสียงของคุณกับ AI ของเรา',
            'home.step1.title': 'อัดเสียงไอ',
            'home.step1.desc': 'กดปุ่มอัดแล้วไอตามธรรมชาติใส่ไมโครโฟนประมาณ 5 วินาที หรือจะอัปโหลดไฟล์เสียงที่มีอยู่ก็ได้',
            'home.step2.title': 'AI วิเคราะห์รูปแบบ',
            'home.step2.desc': 'โมเดล Ensemble CNN+RF ของเราสกัดคุณลักษณะเสียงเชิงลึกจากการไอ — ระดับเสียง จังหวะ และการสั่นพ้อง — เพื่อตรวจหาความผิดปกติของทางเดินหายใจ',
            'home.step3.title': 'รับผลทันที',
            'home.step3.desc': 'รับผลคัดกรองที่ชัดเจน — สุขภาพดี มีอาการ หรือเสี่ยงโควิด-19 — พร้อมคะแนนความมั่นใจภายในไม่ถึง 5 วินาที',
            'home.feat1.title': 'การเรียนรู้แบบ Ensemble CNN+RF',
            'home.feat1.desc': 'ผสานการสกัดคุณลักษณะเชิงลึกด้วย CNN เข้ากับการจำแนกด้วย Random Forest ฝึกบนเสียงไอของผู้ติดโควิด-19 ผู้มีอาการ และผู้มีสุขภาพดี',
            'home.feat2.title': 'ภายใน 5 วินาที',
            'home.feat2.desc': 'ประมวลผลเสียงและสกัดคุณลักษณะแบบเรียลไทม์ ให้ผลแทบจะทันทีในเบราว์เซอร์ของคุณ',
            'home.feat3.title': 'รับประกันความเป็นส่วนตัว',
            'home.feat3.desc': 'เสียงถูกประมวลผลแบบเรียลไทม์และไม่ถูกจัดเก็บ การบันทึกของคุณจะถูกลบทันทีหลังวิเคราะห์',
            'home.feat4.title': 'ใช้ได้ทุกอุปกรณ์',
            'home.feat4.desc': 'รองรับทั้ง iPhone, Android, iPad และเดสก์ท็อปเต็มรูปแบบ ไม่ต้องดาวน์โหลดแอป แค่เปิดเบราว์เซอร์',
            'home.disclaimer.title': 'ไม่ใช่การวินิจฉัยทางการแพทย์',
            'home.disclaimer.desc': 'CoughAI เป็นเครื่องมือคัดกรองด้วย AI สำหรับการตระหนักรู้เบื้องต้นเท่านั้น ไม่ทดแทนคำแนะนำ การวินิจฉัย หรือการรักษาจากแพทย์ หากมีข้อกังวลด้านสุขภาพควรปรึกษาบุคลากรทางการแพทย์เสมอ',
            'home.cta2.title': 'พร้อมตรวจเสียงไอแล้วหรือยัง?',
            'home.cta2.sub': 'ฟรี · ทันที · เป็นส่วนตัว · ไม่ต้องสมัครสมาชิก',
            'home.cta2.btn': 'เริ่มวิเคราะห์ฟรี',
            'home.footer': 'เครื่องมือคัดกรองทางเดินหายใจด้วย AI · ไม่ใช่อุปกรณ์การแพทย์',

            // ════ APP (index.html) ════
            'doc.app.title': 'CoughAI — วิเคราะห์การไอ',
            'app.header.title': 'วิเคราะห์การไอ',
            'app.banner.title': 'บันทึกหรืออัปโหลดเสียงไอ',
            'app.banner.sub': 'วิเคราะห์ด้วย AI · ผลภายใน 5 วินาที',
            'app.group.start': 'เริ่มต้น',
            'app.tile.record.label': 'บันทึกเสียง',
            'app.tile.record.unit': 'วินาที',
            'app.tile.record.desc': 'กดปุ่ม แล้วไอใส่ไมค์',
            'app.wave.idle': 'รอสัญญาณเสียง',
            'app.record.start': 'เริ่มบันทึก',
            'app.divider.or': 'หรือ',
            'app.tile.upload.label': 'อัปโหลด',
            'app.tile.upload.big': 'ไฟล์เสียง',
            'app.drop.hint': 'แตะเพื่อเลือกไฟล์',
            'app.group.result': 'ผลวิเคราะห์',
            'app.loading': 'กำลังวิเคราะห์...',
            'app.result.category': 'การคัดกรอง',
            'app.prob.head': 'ความน่าจะเป็น',
            'app.reco.head': 'คำแนะนำเบื้องต้น',
            'app.reco.disclaimer': '* ผลนี้เป็นการคัดกรองเบื้องต้นด้วย AI ไม่ใช่การวินิจฉัยทางการแพทย์ หากมีอาการน่ากังวล โปรดพบแพทย์เพื่อตรวจยืนยัน',
            'app.btn.hospital': 'รพ. ใกล้ฉัน',
            'app.btn.call': 'โทรปรึกษาแพทย์',
            'app.btn.again': 'วิเคราะห์อีกครั้ง',
            'modal.ok': 'ตกลง',
            'call.title': 'โทรปรึกษา',
            'call.sub': 'เลือกสายด่วนสุขภาพ หรือกรอกเบอร์ รพ. ที่ต้องการ',
            'call.1669': 'เจ็บป่วยฉุกเฉิน (การแพทย์ฉุกเฉิน)',
            'call.1422': 'กรมควบคุมโรค (โควิด / โรคติดต่อ)',
            'call.1330': 'สปสช. (สิทธิรักษา / สอบถามทั่วไป)',
            'call.placeholder': 'เบอร์ รพ. ที่ต้องการโทร',
            'call.btn': 'โทร',
            'call.close': 'ปิด',

            // ── app dynamic ──
            'app.recording': 'กำลังบันทึก...',
            'app.wave.recording': 'กำลังรับเสียง...',
            'app.status.analyzing': 'กำลังวิเคราะห์เสียง...',
            'app.confidence': 'ความมั่นใจ',
            'app.locating': 'กำลังหาตำแหน่ง...',
            'app.error.title': 'ข้อผิดพลาด',
            'app.error.mic': 'ไม่สามารถเข้าถึงไมโครโฟนได้: ',
            'app.error.filetype.title': 'ไฟล์ไม่ถูกต้อง',
            'app.error.filetype.msg': 'กรุณาเลือกไฟล์เสียงเท่านั้น',
            'app.error.nonum.title': 'ยังไม่มีเบอร์',
            'app.error.nonum.msg': 'กรุณากรอกเบอร์โทรก่อน',
            'app.fail.label': 'วิเคราะห์ไม่สำเร็จ',
            'app.fail.retry': 'กรุณาลองใหม่อีกครั้ง',
            'app.fail.reco': 'ไม่สามารถประมวลผลได้ กรุณาอัดเสียงใหม่อีกครั้ง',
            'reco.healthy': 'เสียงไออยู่ในเกณฑ์ปกติ ดูแลสุขภาพ พักผ่อนให้เพียงพอ ดื่มน้ำมาก ๆ',
            'reco.symptomatic': 'พบลักษณะการไอที่ควรเฝ้าระวัง พักผ่อน ดื่มน้ำอุ่น และพบแพทย์หากอาการไม่ดีขึ้นใน 2-3 วัน',
            'reco.covid': 'พบลักษณะการไอที่อาจสัมพันธ์กับโควิด-19 แนะนำตรวจ ATK แยกกักตัว และติดต่อสายด่วน 1422',
            'maps.hospital': 'โรงพยาบาล',
            'maps.hospitalNear': 'โรงพยาบาลใกล้ฉัน',

            // ════ DASHBOARD ════
            'doc.dash.title': 'CoughAI — ประวัติการวิเคราะห์',
            'dash.eyebrow': 'ประวัติการตรวจ',
            'dash.title': 'แดชบอร์ดผลวิเคราะห์',
            'dash.sub': 'ผลการวิเคราะห์เสียงไอทั้งหมดที่อัด/อัปโหลดผ่านเว็บ',
            'dash.refresh': 'รีเฟรช',
            'dash.stat.total': 'ตรวจทั้งหมด',
            'dash.stat.high': 'เสี่ยงสูง',
            'dash.stat.med': 'เฝ้าระวัง',
            'dash.stat.low': 'ปกติ',
            'dash.latest': 'ผลล่าสุด',
            'dash.hist.all': 'ประวัติทั้งหมด',
            'dash.loading': 'กำลังโหลด…',
            'dash.live.connecting': 'กำลังเชื่อมต่อ…',
            'dash.live.on': 'เชื่อมต่อแล้ว',
            'dash.live.off': 'เชื่อมต่อไม่ได้',
            'dash.hist.time': 'เวลา',
            'dash.hist.result': 'ผลวิเคราะห์',
            'dash.hist.level': 'ระดับ',
            'dash.empty': 'ยังไม่มีประวัติ — ลองอัดหรืออัปโหลดเสียงไอที่หน้า App',
            'dash.empty.cta': 'ไปหน้า App →',
            'time.justnow': 'เมื่อสักครู่',
            'time.sec': 'วินาทีที่แล้ว',
            'time.min': 'นาทีที่แล้ว',
            'time.hour': 'ชั่วโมงที่แล้ว',
        },

        en: {
            // ── nav ──
            'nav.home': 'Home',
            'nav.app': 'App',
            'nav.dashboard': 'Dashboard',

            // ── labels / risk ──
            'label.healthy': 'Healthy',
            'label.covid': 'COVID-19',
            'label.symptomatic': 'Symptomatic',
            'risk.HIGH': 'High risk',
            'risk.MEDIUM': 'Watch',
            'risk.LOW': 'Normal',

            // ════ HOMEPAGE ════
            'doc.home.title': 'CoughAI — Respiratory Health Screening',
            'home.cta': 'Start Analysis →',
            'home.badge': 'AI-Powered · Real-Time Analysis',
            'home.headline': 'Turn your cough into<br><em>early health insight</em>',
            'home.sub': 'AI analyzes your cough in seconds to assess<br class="desktop-br"> respiratory risk — no clinic visit needed.',
            'home.start': 'Start Analysis',
            'home.how': 'How it works',
            'home.trust.privacy': 'Privacy First',
            'home.trust.fast': 'Results in <5s',
            'home.trust.noupload': 'No Upload Required',
            'home.visual.analyzing': 'Analyzing audio pattern',
            'home.visual.diagnosis': 'Diagnosis',
            'home.chip.sound': 'Sound detected',
            'home.chip.model': 'AI model ready',
            'home.how.eyebrow': 'Simple process',
            'home.how.title': 'Three steps to insight',
            'home.how.sub': 'No doctor visit. No lab test. Just your voice and our AI.',
            'home.step1.title': 'Record your cough',
            'home.step1.desc': 'Tap the record button and cough naturally into your device microphone for 5 seconds. Or upload an existing audio file.',
            'home.step2.title': 'AI analyzes patterns',
            'home.step2.desc': 'Our CNN+RF ensemble extracts deep acoustic features from your cough — pitch, rhythm, and resonance — to detect respiratory anomalies.',
            'home.step3.title': 'Get instant results',
            'home.step3.desc': 'Receive a clear classification — Healthy, Symptomatic, or COVID-19 risk — with confidence scores in under 5 seconds.',
            'home.feat1.title': 'Ensemble Learning CNN+RF',
            'home.feat1.desc': 'Combines CNN deep feature extraction with Random Forest classification, trained on COVID-19 positive, symptomatic, and healthy cough recordings.',
            'home.feat2.title': 'Under 5 Seconds',
            'home.feat2.desc': 'Real-time audio processing and feature extraction delivers results almost instantly, directly in your browser.',
            'home.feat3.title': 'Privacy Guaranteed',
            'home.feat3.desc': 'Audio is processed in real time and never stored. Your recordings are deleted immediately after analysis.',
            'home.feat4.title': 'Works Everywhere',
            'home.feat4.desc': 'Fully responsive for iPhone, Android, iPad, and desktop. No app download needed — just open your browser.',
            'home.disclaimer.title': 'Not a medical diagnosis',
            'home.disclaimer.desc': 'CoughAI is an AI-powered screening tool designed for early awareness only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for health concerns.',
            'home.cta2.title': 'Ready to check your cough?',
            'home.cta2.sub': 'Free · Instant · Private · No account needed',
            'home.cta2.btn': 'Start Free Analysis',
            'home.footer': 'AI-powered respiratory screening tool · Not a medical device',

            // ════ APP ════
            'doc.app.title': 'CoughAI — Cough Analysis',
            'app.header.title': 'Cough Analysis',
            'app.banner.title': 'Record or upload a cough',
            'app.banner.sub': 'AI analysis · results in 5 seconds',
            'app.group.start': 'Get started',
            'app.tile.record.label': 'Record',
            'app.tile.record.unit': 'seconds',
            'app.tile.record.desc': 'Tap, then cough into the mic',
            'app.wave.idle': 'Waiting for audio',
            'app.record.start': 'Start recording',
            'app.divider.or': 'or',
            'app.tile.upload.label': 'Upload',
            'app.tile.upload.big': 'Audio file',
            'app.drop.hint': 'Tap to choose a file',
            'app.group.result': 'Result',
            'app.loading': 'Analyzing...',
            'app.result.category': 'Screening',
            'app.prob.head': 'Probabilities',
            'app.reco.head': 'Initial guidance',
            'app.reco.disclaimer': '* This is an AI-based preliminary screening, not a medical diagnosis. If you have concerning symptoms, please see a doctor for confirmation.',
            'app.btn.hospital': 'Hospitals nearby',
            'app.btn.call': 'Call a doctor',
            'app.btn.again': 'Analyze again',
            'modal.ok': 'OK',
            'call.title': 'Call for advice',
            'call.sub': 'Pick a health hotline or enter a hospital number',
            'call.1669': 'Medical emergency (EMS)',
            'call.1422': 'Disease Control (COVID / infectious)',
            'call.1330': 'NHSO (coverage / general inquiries)',
            'call.placeholder': 'Hospital number to call',
            'call.btn': 'Call',
            'call.close': 'Close',

            // ── app dynamic ──
            'app.recording': 'Recording...',
            'app.wave.recording': 'Listening...',
            'app.status.analyzing': 'Analyzing audio...',
            'app.confidence': 'Confidence',
            'app.locating': 'Locating...',
            'app.error.title': 'Error',
            'app.error.mic': 'Cannot access microphone: ',
            'app.error.filetype.title': 'Invalid file',
            'app.error.filetype.msg': 'Please select an audio file only',
            'app.error.nonum.title': 'No number',
            'app.error.nonum.msg': 'Please enter a phone number first',
            'app.fail.label': 'Analysis failed',
            'app.fail.retry': 'Please try again',
            'app.fail.reco': 'Could not process. Please record again.',
            'reco.healthy': 'Your cough is within the normal range. Stay healthy, rest well, and drink plenty of water.',
            'reco.symptomatic': 'Patterns worth monitoring were detected. Rest, drink warm fluids, and see a doctor if it does not improve within 2-3 days.',
            'reco.covid': 'Patterns possibly linked to COVID-19 were detected. Consider an ATK test, self-isolate, and contact the hotline 1422.',
            'maps.hospital': 'hospital',
            'maps.hospitalNear': 'hospital near me',

            // ════ DASHBOARD ════
            'doc.dash.title': 'CoughAI — Analysis History',
            'dash.eyebrow': 'Test history',
            'dash.title': 'Analysis dashboard',
            'dash.sub': 'All cough analyses recorded/uploaded through the web',
            'dash.refresh': 'Refresh',
            'dash.stat.total': 'Total',
            'dash.stat.high': 'High risk',
            'dash.stat.med': 'Watch',
            'dash.stat.low': 'Normal',
            'dash.latest': 'Latest result',
            'dash.hist.all': 'All history',
            'dash.loading': 'Loading…',
            'dash.live.connecting': 'Connecting…',
            'dash.live.on': 'Connected',
            'dash.live.off': 'Disconnected',
            'dash.hist.time': 'Time',
            'dash.hist.result': 'Result',
            'dash.hist.level': 'Level',
            'dash.empty': 'No history yet — record or upload a cough on the App page',
            'dash.empty.cta': 'Go to App →',
            'time.justnow': 'just now',
            'time.sec': 's ago',
            'time.min': 'min ago',
            'time.hour': 'h ago',
        },
    };

    let lang = DEFAULT;
    try {
        const saved = localStorage.getItem(STORE_KEY);
        if (saved === 'th' || saved === 'en') lang = saved;
    } catch (e) {}

    function t(key) {
        return (DICT[lang] && DICT[lang][key]) || DICT.th[key] || key;
    }

    function apply() {
        const root = document.documentElement;
        root.setAttribute('lang', lang);

        document.querySelectorAll('[data-i18n]').forEach(el => {
            el.textContent = t(el.getAttribute('data-i18n'));
        });
        document.querySelectorAll('[data-i18n-html]').forEach(el => {
            el.innerHTML = t(el.getAttribute('data-i18n-html'));
        });
        document.querySelectorAll('[data-i18n-ph]').forEach(el => {
            el.setAttribute('placeholder', t(el.getAttribute('data-i18n-ph')));
        });

        // ป้ายปุ่มสลับภาษา → แสดงภาษาปลายทาง
        const tgt = lang === 'th' ? 'EN' : 'ไทย';
        document.querySelectorAll('.lang-label').forEach(el => { el.textContent = tgt; });

        // หัวเรื่องเอกสารตามหน้า
        const titleKey = root.getAttribute('data-doc-title');
        if (titleKey) document.title = t(titleKey);

        // แจ้ง JS อื่น ๆ ให้ render ใหม่
        document.dispatchEvent(new CustomEvent('coughai:langchange', { detail: { lang } }));
    }

    function set(next) {
        if (next !== 'th' && next !== 'en') return;
        lang = next;
        try { localStorage.setItem(STORE_KEY, lang); } catch (e) {}
        apply();
    }

    // ── public API ──
    window.t = t;
    window.CoughLang = {
        get: () => lang,
        set,
        toggle: () => set(lang === 'th' ? 'en' : 'th'),
    };
    window.toggleLang = () => window.CoughLang.toggle();

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', apply);
    } else {
        apply();
    }
})();
