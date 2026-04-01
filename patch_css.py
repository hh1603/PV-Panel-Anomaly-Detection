path = 'scripts/web_demo.py'
with open(path, encoding='utf-8') as f:
    src = f.read()

old = '    .two-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;}}'
new = '    .two-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;}}\n    .compare-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0;}}\n    .compare-item{{display:grid;gap:8px;}}\n    @media(max-width:600px){{.compare-grid{{grid-template-columns:1fr;}}}}'

if old in src:
    src = src.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    print('CSS patched OK')
else:
    print('pattern not found')
