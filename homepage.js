// CoughAI — homepage.js
// Scroll reveal + nav scroll effect + result cycling

document.addEventListener('DOMContentLoaded', () => {

    // ── Nav scroll effect ──────────────────────────────────
    const nav = document.getElementById('nav');
    const onScroll = () => {
        if (window.scrollY > 20) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    };
    window.addEventListener('scroll', onScroll, { passive: true });

    // ── Scroll reveal ──────────────────────────────────────
    const revealEls = document.querySelectorAll('.scroll-reveal, .scroll-reveal-stagger');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    });

    revealEls.forEach(el => observer.observe(el));

    // ── Smooth scroll for anchor links ────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ── Waveform animation randomization ─────────────────
    const wbars = document.querySelectorAll('.wbar');
    const randomizeWave = () => {
        wbars.forEach(bar => {
            const h = 20 + Math.random() * 75;
            bar.style.setProperty('--h', `${h}%`);
        });
    };
    setInterval(randomizeWave, 1500);

    // ── Result cycling (Healthy / Symptomatic / COVID-19) ─
    const results = [
        {
            type: 'healthy',
            label: 'Healthy',
            conf: '96.4%',
            iconSvg: '<polyline points="20 6 9 17 4 12"/>',
            bars: [
                { label: 'Healthy',     pct: 96.4, color: '#22c55e' },
                { label: 'Symptomatic', pct: 3.1,  color: '#eab308' },
                { label: 'COVID-19',    pct: 0.5,  color: '#ef4444' },
            ]
        },
        {
            type: 'symptomatic',
            label: 'Symptomatic',
            conf: '78.2%',
            iconSvg: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
            bars: [
                { label: 'Healthy',     pct: 14.3, color: '#22c55e' },
                { label: 'Symptomatic', pct: 78.2, color: '#eab308' },
                { label: 'COVID-19',    pct: 7.5,  color: '#ef4444' },
            ]
        },
        {
            type: 'covid',
            label: 'COVID-19',
            conf: '84.7%',
            iconSvg: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
            bars: [
                { label: 'Healthy',     pct: 6.8,  color: '#22c55e' },
                { label: 'Symptomatic', pct: 8.5,  color: '#eab308' },
                { label: 'COVID-19',    pct: 84.7, color: '#ef4444' },
            ]
        },
    ];

    let currentIdx = 0;

    const rpIcon  = document.querySelector('.rp-icon');
    const rpValue = document.querySelector('.rp-value');
    const rpConf  = document.querySelector('.rp-conf');
    const rpFills = document.querySelectorAll('.rp-fill');
    const rpPcts  = document.querySelectorAll('.rp-bar-row span:last-child');

    if (rpIcon && rpValue) {
        const cycleResult = () => {
            currentIdx = (currentIdx + 1) % results.length;
            const r = results[currentIdx];

            // Flip animation on icon
            rpIcon.classList.add('flip');

            setTimeout(() => {
                // Update icon
                rpIcon.className = `rp-icon ${r.type}`;
                rpIcon.querySelector('svg').innerHTML = r.iconSvg;
                rpIcon.classList.remove('flip');

                // Update label
                rpValue.className = `rp-value ${r.type}`;
                rpValue.textContent = r.label;

                // Update confidence
                rpConf.textContent = r.conf;

                // Update bars
                r.bars.forEach((b, i) => {
                    if (rpFills[i]) {
                        rpFills[i].style.width = `${b.pct}%`;
                        rpFills[i].style.background = b.color;
                    }
                    if (rpPcts[i]) {
                        rpPcts[i].textContent = `${b.pct}%`;
                    }
                });
            }, 200);
        };

        // Cycle every 3.5 seconds
        setInterval(cycleResult, 3500);
    }
});
