Here are **lightweight, language-agnostic Python packages for sentence completion detection**, along with FastAPI + Docker compatibility considerations:

---

### ✅ Key Criteria:

* **Lightweight** (minimal dependencies, fast startup)
* **Language-agnostic** (not tied only to English)
* **Usable in FastAPI + Docker**
* **Detect sentence completion** (e.g., if a sentence ends naturally or is cut off)

---

## 🔹 Recommended Packages

### 1. **`langdetect` or `langid`** *(for language detection)*

* Useful as a **first step** to detect the language.
* `langdetect`: Pure Python, lightweight
* `langid`: Faster and more consistent across domains

```bash
pip install langdetect
```

---

### 2. **`spacy` with lightweight models**

* Language-agnostic via multilingual models
* Use it to **detect sentence boundaries**
* Lightweight model: `xx_sent_ud_sm`

```bash
pip install spacy
python -m spacy download xx_sent_ud_sm
```

#### Code Snippet:

```python
import spacy

nlp = spacy.load("xx_sent_ud_sm")
doc = nlp("This is a complete sentence. This is not")

for sent in doc.sents:
    print(sent.text)
```

---

### 3. **`textstat`**

* Lightweight way to measure readability and sentence structure
* Can help judge if a sentence is "complete" via structure complexity

```bash
pip install textstat
```

---

### 4. **`transformers` with `distilBERT`** *(Optional but heavier)*

* Can classify sentence endings using masked language modeling (MLM)
* Use a very small model like `distilbert-base-multilingual-cased` for language-agnostic usage

> ✅ Use only if basic methods don’t meet your quality requirements

---

## 🐳 Docker + FastAPI Setup

### **`Dockerfile` (lightweight setup)**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt` example

```
fastapi
uvicorn
spacy
langdetect
textstat
```

---

### **FastAPI Example: `main.py`**

```python
from fastapi import FastAPI
import spacy

app = FastAPI()
nlp = spacy.load("xx_sent_ud_sm")

@app.post("/is_complete/")
def is_sentence_complete(text: str):
    doc = nlp(text)
    sentences = list(doc.sents)
    if len(sentences) == 1 and not text.strip().endswith(('.', '?', '!')):
        return {"complete": False}
    return {"complete": True}
```

---

## 🧠 Bonus Tip

If you want **ML-based confidence scoring** for sentence completion, use `transformers` in combination with masked language modeling (e.g., appending a mask at the end and checking predictability). Let me know if you want an example of that too.

---

Would you like a GitHub-ready example repo or Docker Compose setup as well?
