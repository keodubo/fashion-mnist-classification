"""Bundle the HTML, CSS, JavaScript and real project data into one offline file."""
from pathlib import Path
import base64
import json

HERE = Path(__file__).resolve().parent
template = (HERE / "template.html").read_text()
payload = json.loads((HERE / "data.json").read_text())
payload["interpretation"] = json.loads((HERE / "interpretation.json").read_text())
payload["notes"] = json.loads((HERE / "speaker-notes.json").read_text())
payload["scene_notes"] = json.loads((HERE / "scene-notes.json").read_text())
data = json.dumps(payload, separators=(",", ":"))
assert json.loads(data)["demos"]
assert "</script" not in data.lower()
chart = base64.b64encode((HERE / "coat-shap-slide.png").read_bytes()).decode("ascii")
template = template.replace("/* SHAP IMAGE */", "data:image/png;base64," + chart)
html = template.replace("/* STYLES */", (HERE / "styles.css").read_text())
html = html.replace("/* DATA */", data).replace("/* SCRIPT */", (HERE / "interpretation.js").read_text() + "\n" + (HERE / "app.js").read_text())
assert "/* DATA */" not in html and "/* SCRIPT */" not in html
(HERE / "index.html").write_text(html)
output = HERE.parent / "output/fashion-mnist-presentation.html"
output.write_text(html)
print(f"Built self-contained HTML: {HERE / 'index.html'} ({len(html.encode()):,} bytes)")
print(f"Shareable file: {output}")
