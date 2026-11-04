"""Bundle the HTML, CSS, JavaScript and real project data into one offline file."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
template = (HERE / "template.html").read_text()
data = (HERE / "data.json").read_text()
assert json.loads(data)["demos"]
assert "</script" not in data.lower()
html = template.replace("/* STYLES */", (HERE / "styles.css").read_text())
html = html.replace("/* DATA */", data).replace("/* SCRIPT */", (HERE / "app.js").read_text())
assert "/* DATA */" not in html and "/* SCRIPT */" not in html
(HERE / "index.html").write_text(html)
output = HERE.parent / "output/fashion-mnist-interactive-v1.html"
output.write_text(html)
print(f"Built self-contained HTML: {HERE / 'index.html'} ({len(html.encode()):,} bytes)")
print(f"Shareable file: {output}")
