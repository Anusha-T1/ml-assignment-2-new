"""Render the submission PDF: cover page with the mandatory links, then the README."""

from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/Anusha-T1/ml-assignment-2-new"
STREAMLIT_URL = "https://YOUR-APP.streamlit.app"
SCREENSHOT = HERE / "bits_lab_screenshot.png"

readme_html = markdown.markdown(
    (HERE / "README.md").read_text(),
    extensions=["tables", "fenced_code"],
)

shot_block = (
    f'<img src="{SCREENSHOT.name}" alt="BITS Virtual Lab execution">'
    if SCREENSHOT.exists()
    else '<div class="placeholder">[ Paste your BITS Virtual Lab screenshot here ]</div>'
)

CSS = """
@page { size: A4; margin: 18mm 15mm; @bottom-center {
  content: counter(page); font-size: 8pt; color: #888; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; color: #6b1030; border-bottom: 2px solid #6b1030;
     padding-bottom: 4px; }
h2 { font-size: 12pt; color: #6b1030; margin-top: 16px; }
h3 { font-size: 10.5pt; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 7.8pt; }
th, td { border: 0.6pt solid #b9b9b9; padding: 4px 5px; text-align: left;
         vertical-align: top; }
th { background: #f3e8ec; font-weight: bold; }
tr:nth-child(even) td { background: #fbfbfb; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 7.5pt;
            background: #f5f5f5; }
pre { padding: 7px; border-left: 2.5pt solid #6b1030; white-space: pre-wrap; }
a { color: #6b1030; }
.cover { text-align: center; margin-bottom: 22px; }
.cover .sub { color: #555; font-size: 10pt; }
.links { border: 0.8pt solid #6b1030; border-radius: 4px; padding: 10px 14px;
         margin: 14px 0; font-size: 9.5pt; }
.links div { margin: 5px 0; }
.placeholder { border: 1pt dashed #999; padding: 46px 10px; text-align: center;
               color: #888; font-style: italic; }
img { max-width: 100%; border: 0.6pt solid #ccc; }
.pagebreak { page-break-after: always; }
"""

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="cover">
  <h1>Machine Learning &mdash; Assignment 2</h1>
  <div class="sub">Work Integrated Learning Programmes Division &middot;
  M.Tech (AIML/DSE) &middot; BITS Pilani</div>
  <div class="sub"><b>Cellar Check</b> &mdash; premium wine classification from
  physicochemical data</div>
</div>

<h2>1. GitHub repository link</h2>
<div class="links">
  <div><a href="{GITHUB_URL}">{GITHUB_URL}</a></div>
  <div>Contains: <code>app.py</code>, <code>requirements.txt</code>,
  <code>README.md</code>, <code>test_data.csv</code>, <code>model/</code>
  (training script + all six saved models), <code>data/</code> (raw UCI CSVs).</div>
</div>

<h2>2. Live Streamlit app link</h2>
<div class="links">
  <div><a href="{STREAMLIT_URL}">{STREAMLIT_URL}</a></div>
  <div>Deployed on Streamlit Community Cloud. Opens an interactive frontend with
  CSV upload, a six-way model dropdown, live evaluation metrics, a confusion
  matrix, an ROC curve and a full classification report.</div>
</div>

<h2>3. Screenshot &mdash; execution on BITS Virtual Lab</h2>
{shot_block}

<div class="pagebreak"></div>

<h2>4. README.md contents</h2>
{readme_html}
</body></html>"""

out = HERE / "ML_Assignment2_Submission.pdf"
HTML(string=html, base_url=str(HERE)).write_pdf(out)
print("wrote", out)
