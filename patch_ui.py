import re

path = 'scripts/web_demo.py'
with open(path, encoding='utf-8') as f:
    src = f.read()

# ---- new render_layout ----
NEW_FUNC = r'''def render_layout(title: str, body: str) -> bytes:
    cards = "".join(
        f"""
        <article class=\"pipeline-card\">
          <p class=\"section-label section-label-green\">{html.escape(item['modality'])}</p>
          <h3>{html.escape(item['pipeline'])}</h3>
          <p><span class=\"tag\">{html.escape(item['task'])}</span></p>
          <p class=\"path\">{html.escape(item['weights'])}</p>
        </article>
        """
        for item in pipeline_overview()
    )

    page = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)} — 光伏异常检测</title>
  <style>
    :root {{
      --bg:#0d1117;--surface:#161b22;--surface2:#21262d;
      --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.13);
      --ink:#e6edf3;--muted:#7d8590;--faint:#484f58;
      --accent:#3b9eff;--accent-dim:rgba(59,158,255,0.12);
      --green:#3fb950;--green-dim:rgba(63,185,80,0.12);
      --orange:#f0883e;--danger:#f85149;
      --r:10px;--shadow:0 8px 32px rgba(0,0,0,0.45);
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    body{{
      min-height:100vh;color:var(--ink);background:var(--bg);
      font-family:-apple-system,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif;
      font-size:15px;line-height:1.65;
      background-image:radial-gradient(ellipse 80% 35% at 50% -5%,rgba(59,158,255,0.07),transparent);
    }}
    a{{color:var(--accent);text-decoration:none;}}
    a:hover{{text-decoration:underline;}}
    nav{{
      position:sticky;top:0;z-index:100;
      height:52px;padding:0 20px;
      display:flex;align-items:center;gap:10px;
      background:rgba(13,17,23,0.88);backdrop-filter:blur(14px);
      border-bottom:1px solid var(--border);
    }}
    .nav-logo{{
      width:28px;height:28px;border-radius:7px;flex-shrink:0;
      background:linear-gradient(135deg,#3b9eff,#3fb950);
      display:flex;align-items:center;justify-content:center;font-size:15px;
    }}
    .nav-title{{font-weight:700;font-size:14px;color:var(--ink);}}
    .nav-pill{{
      padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;
      background:var(--green-dim);color:var(--green);border:1px solid rgba(63,185,80,0.25);
    }}
    .nav-pill::before{{content:"\u25cf ";font-size:8px;}}
    .nav-space{{flex:1;}}
    .nav-hint{{font-size:12px;color:var(--muted);}}
    .shell{{width:min(1160px,calc(100% - 32px));margin:28px auto 72px;display:grid;gap:20px;}}
    .hero{{
      position:relative;overflow:hidden;
      background:linear-gradient(135deg,var(--surface) 0%,#0b1628 100%);
      border:1px solid var(--border2);border-radius:14px;
      padding:36px 32px;box-shadow:var(--shadow);
    }}
    .hero::before{{
      content:"";position:absolute;inset:0;
      background:radial-gradient(circle at 90% 10%,rgba(59,158,255,0.08),transparent 55%);
      pointer-events:none;
    }}
    .status-badge{{
      display:inline-flex;align-items:center;gap:6px;
      padding:3px 11px;border-radius:20px;font-size:12px;font-weight:600;
      background:var(--green-dim);color:var(--green);border:1px solid rgba(63,185,80,0.28);
      margin-bottom:16px;
    }}
    .status-dot{{width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;box-shadow:0 0 6px var(--green);}}
    .hero h1{{font-size:clamp(22px,3.5vw,34px);font-weight:700;color:var(--ink);margin-bottom:10px;line-height:1.2;}}
    .hero p{{color:var(--muted);font-size:14px;max-width:520px;}}
    .two-col{{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start;}}
    @media(max-width:860px){{.two-col{{grid-template-columns:1fr;}}}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:22px;box-shadow:var(--shadow);}}
    .section-label{{font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;display:block;}}
    .section-label-green{{color:var(--green);}}
    .section-label-orange{{color:var(--orange);}}
    .form-grid{{display:grid;gap:16px;}}
    .field{{display:grid;gap:6px;}}
    .field label{{font-size:13px;font-weight:600;color:var(--muted);}}
    select,input[type=file]{{
      width:100%;padding:10px 14px;
      background:var(--surface2);color:var(--ink);
      border:1px solid var(--border2);border-radius:8px;
      font:inherit;font-size:14px;transition:border-color .15s;
    }}
    select:focus,input[type=file]:focus{{outline:none;border-color:var(--accent);}}
    .btn{{
      display:inline-flex;align-items:center;justify-content:center;gap:7px;
      padding:10px 18px;border-radius:8px;border:none;
      font:inherit;font-size:14px;font-weight:600;
      cursor:pointer;transition:opacity .15s,transform .12s;text-decoration:none;
    }}
    .btn:hover{{opacity:.85;transform:translateY(-1px);text-decoration:none;}}
    .btn-primary{{background:var(--accent);color:#fff;width:100%;padding:12px;}}
    .btn-ghost{{background:var(--surface2);color:var(--ink);border:1px solid var(--border2);}}
    .tips-list{{list-style:none;display:grid;gap:7px;margin-top:14px;}}
    .tips-list li{{
      font-size:13px;color:var(--muted);padding:9px 13px;
      background:var(--surface2);border-radius:8px;border-left:3px solid var(--border2);
    }}
    .pipeline-card{{padding:14px;border-radius:9px;background:var(--surface2);border:1px solid var(--border);margin-bottom:10px;}}
    .pipeline-card h3{{font-size:13px;font-weight:700;color:var(--ink);margin-bottom:5px;}}
    .tag{{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;background:var(--accent-dim);color:var(--accent);border:1px solid rgba(59,158,255,.2);}}
    .path{{word-break:break-all;font-size:11px;color:var(--faint);margin-top:5px;}}
    .result-image{{width:100%;display:block;border-radius:10px;border:1px solid var(--border);margin:16px 0;}}
    .kv-list{{list-style:none;display:grid;gap:8px;}}
    .kv-list li{{
      display:flex;justify-content:space-between;align-items:baseline;
      padding:9px 13px;background:var(--surface2);border-radius:8px;
      border:1px solid var(--border);font-size:13px;gap:10px;
    }}
    .kv-list li strong{{color:var(--muted);font-weight:600;flex-shrink:0;}}
    .det-list{{list-style:none;display:grid;gap:8px;}}
    .det-list li{{
      padding:10px 13px;background:var(--surface2);
      border-radius:8px;border:1px solid var(--border);
      font-size:13px;color:var(--ink);border-left:3px solid var(--orange);
    }}
    .action-row{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0;}}
    .report-card{{margin:16px 0;padding:18px;border-radius:10px;background:var(--surface2);border:1px solid var(--border);}}
    .two-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;}}
    @media(max-width:600px){{.two-grid{{grid-template-columns:1fr;}}}}
    pre{{margin:0;white-space:pre-wrap;word-break:break-word;background:#0d1f1a;color:#adbfb8;border-radius:10px;padding:16px;overflow:auto;font-size:12px;border:1px solid var(--border);}}
    .error{{padding:12px 14px;border-radius:8px;background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,.3);color:#f85149;font-size:14px;margin-bottom:12px;}}
    .metric-list,.detection-list{{list-style:none;display:grid;gap:8px;}}
    .metric-list li,.detection-list li{{padding:9px 13px;background:var(--surface2);border-radius:8px;border:1px solid var(--border);font-size:13px;color:var(--ink);}}
    .no-print{{}}
    @media(max-width:720px){{
      .shell{{width:calc(100% - 24px);}}
      .hero{{padding:22px 18px;}}
      .btn{{width:100%;}}
      .action-row{{flex-direction:column;}}
    }}
    @media print{{
      body{{background:#fff;color:#000;}}
      nav,.hero,.action-row.no-print,.raw-json.no-print{{display:none!important;}}
      .card{{box-shadow:none;border:1px solid #ddd;background:#fff;}}
      .two-col{{grid-template-columns:1fr;}}
    }}
  </style>
</head>
<body>
  <nav>
    <div class=\"nav-logo\">\u2600</div>
    <span class=\"nav-title\">光伏异常检测系统</span>
    <span class=\"nav-pill\">运行中</span>
    <div class=\"nav-space\"></div>
    <span class=\"nav-hint\">本地推理 · CPU</span>
  </nav>
  <main class=\"shell\">
    <section class=\"hero no-print\">
      <div class=\"status-badge\"><span class=\"status-dot\"></span>系统就绪</div>
      <h1>光伏板异常检测<br>可视化平台</h1>
      <p>上传可见光或红外图像，系统自动选择对应模型进行推理，结果实时展示并可导出报告。</p>
    </section>
    <div class=\"two-col\">
      <section class=\"card\">{body}</section>
      <aside class=\"card no-print\">
        <span class=\"section-label section-label-green\">当前激活流水线</span>
        <div>{cards}</div>
      </aside>
    </div>
  </main>
</body>
</html>
"""
    return page.encode('utf-8')
'''

start = src.index('def render_layout(')
end = src.index('\ndef render_home_page(')
src = src[:start] + NEW_FUNC + '\n' + src[end:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print('Done')
