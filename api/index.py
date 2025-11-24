from io import BytesIO
import os
from pathlib import Path

from asgiref.wsgi import WsgiToAsgi
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from PyPDF2 import PdfReader

BASE_DIR = Path(__file__).resolve().parents[1]
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
MAX_FILE_BYTES = 5 * 1024 * 1024  # limit uploads to 5 MB

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Hot Mess Coach</title>
    <style>
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #0f172a;
        color: #e2e8f0;
        margin: 0;
        padding: 2rem;
      }
      main {
        max-width: 720px;
        margin: 0 auto;
        background: #1e293b;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 10px 40px rgb(15 23 42 / 0.6);
      }
      h1 {
        margin-top: 0;
      }
      label {
        display: block;
        margin-top: 1.25rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
      }
      textarea,
      input[type="text"],
      input[type="file"] {
        width: 100%;
        margin-top: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #475569;
        background: #0f172a;
        color: #e2e8f0;
        resize: vertical;
      }
      button {
        margin-top: 1.5rem;
        padding: 0.85rem 1.5rem;
        border-radius: 9999px;
        border: none;
        font-weight: 600;
        letter-spacing: 0.05em;
        cursor: pointer;
        color: white;
        background: linear-gradient(120deg, #f97316, #ef4444, #a855f7);
        box-shadow: 0 10px 25px rgb(233 63 63 / 0.3);
      }
      .response {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 1rem;
        background: rgb(79 70 229 / 0.1);
        border: 1px solid rgb(79 70 229 / 0.4);
        white-space: pre-wrap;
      }
      .doc-preview {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 0.75rem;
        background: #0f172a;
        border: 1px solid #475569;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>🔥 Hot Mess Coach</h1>
      <p>Your supportive mini mental coach powered by gpt-4o-mini.</p>

      <form method="post" enctype="multipart/form-data">
        <label for="user_msg">How are you feeling today?</label>
        <textarea
          id="user_msg"
          name="user_msg"
          rows="4"
          placeholder="I feel like a hot mess today..."
          required
        >{{ user_msg }}</textarea>

        <label for="file">Upload PDF or CSV (optional)</label>
        <input id="file" name="file" type="file" accept=".pdf,.csv" />

        <button type="submit">Coach me</button>
      </form>

      {% if doc_preview %}
      <section class="doc-preview">
        <strong>Document preview:</strong>
        <pre>{{ doc_preview }}</pre>
      </section>
      {% endif %}

      {% if reply %}
      <section class="response">
        <h2>💬 Coach says:</h2>
        <p>{{ reply }}</p>
      </section>
      {% endif %}

      {% if error %}
      <section class="response" style="border-color:#f87171;background:rgb(248 113 113 / 0.1);">
        <strong>Error:</strong>
        <p>{{ error }}</p>
      </section>
      {% endif %}
    </main>
  </body>
</html>
"""


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as exc:
        return f"[PDF extraction error]: {exc}"


def extract_csv_text(file_storage) -> str:
    """Turn a CSV upload into a printable string."""
    try:
        df = pd.read_csv(file_storage)
        return df.to_string()
    except Exception as exc:
        return f"[CSV parsing error]: {exc}"


@app.route("/", methods=["GET", "POST"])
def hot_mess_coach():
    user_msg = ""
    reply = None
    doc_preview = None
    error = None

    if request.method == "POST":
        user_msg = request.form.get("user_msg", "").strip()
        uploaded_file = request.files.get("file")

        uploaded_content = None
        if uploaded_file and uploaded_file.filename:
            size = uploaded_file.content_length
            if size is None:
                current_pos = uploaded_file.stream.tell()
                uploaded_file.stream.seek(0, os.SEEK_END)
                size = uploaded_file.stream.tell()
                uploaded_file.stream.seek(current_pos)

            if size and size > MAX_FILE_BYTES:
                error = "Files must be 5 MB or smaller."
            else:
                uploaded_file.stream.seek(0)
                mime = uploaded_file.mimetype
                if mime == "text/csv":
                    doc_preview = extract_csv_text(uploaded_file)
                elif mime in ("application/pdf", "application/octet-stream"):
                    doc_preview = extract_pdf_text(uploaded_file.read())
                else:
                    error = "Only PDF or CSV files are supported."

                uploaded_content = doc_preview

        if not error and user_msg:
            try:
                system_prompt = "You are a supportive mental coach who helps overwhelmed people feel calmer."
                if uploaded_content:
                    system_prompt += (
                        "\n\nThe user has also uploaded a document. Here is the content:\n"
                        f"{uploaded_content}"
                    )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                reply = response.choices[0].message.content
            except Exception as exc:
                error = f"Something went wrong while contacting OpenAI: {exc}"

    return render_template_string(
        PAGE_TEMPLATE,
        user_msg=user_msg,
        reply=reply,
        doc_preview=doc_preview,
        error=error,
    )


asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)