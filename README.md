# Hasha — Python backend

Server-side version of the Hasha AI Teacher teaching loop: lesson planning,
RAG-grounded concept explanations, adaptive evaluation, quizzes, and reports.
Same logic as the browser prototype, but running behind a real API so your
Anthropic key never sits in client-side JavaScript.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY
python app.py
```

Server runs at `http://localhost:5000`.

## Endpoints

All endpoints accept and return JSON.

### `POST /api/plan`
Generate the ordered concept list for a lesson.
```json
{"topic": "Ohm's Law", "level": "Beginner", "time": "20 minutes",
 "language": "English", "material": "optional pasted chapter text"}
```

### `POST /api/concept`
Generate the explanation, subject-aware visual spec, and checkpoint question
for one concept. Retrieval runs server-side with TF-IDF over `material`.
```json
{"topic": "Ohm's Law", "concept_title": "What is resistance?",
 "level": "Beginner", "language": "English", "material": "...",
 "alt_analogy": false}
```
Set `"alt_analogy": true` on a re-teach pass so Claude is told to use a
different explanation than the first attempt.

### `POST /api/evaluate`
Check a student's answer and detect misconceptions.
```json
{"question_prompt": "...", "correct_answer": "...",
 "student_answer": "...", "language": "English"}
```
Returns `{"correct": bool, "misconception": "...", "feedback": "...",
"nextAction": "continue" | "reteach"}`.

### `POST /api/quiz`
Generate a 4-question final quiz covering the lesson's concepts.
```json
{"topic": "Ohm's Law", "concept_titles": ["...", "..."],
 "level": "Beginner", "language": "English"}
```

### `POST /api/report`
Generate the closing learning report.
```json
{"topic": "Ohm's Law", "score": 75, "missed_questions": ["..."],
 "correct_questions": ["..."], "profile_weak": ["..."], "language": "English"}
```

### `GET /api/profile/<student_id>` / `POST /api/profile/<student_id>`
Read/write a learner profile (`{"strong": [...], "weak": [...], "history": [...]}`),
persisted to `profiles.json` in this folder. Swap for a real database for a
production submission.

## Files

- `app.py` — Flask routes, one per step of the teaching loop
- `rag.py` — chunking + TF-IDF retrieval (swap for embeddings + a vector DB later)
- `claude_client.py` — Anthropic SDK wrapper with defensive JSON parsing
- `requirements.txt`, `.env.example`

## Wiring up the frontend

Point the HTML prototype's `fetch()` calls at `http://localhost:5000/api/...`
instead of `https://api.anthropic.com/v1/messages` directly, and drop the
in-browser system-prompt strings (the backend now owns those). This is the
architecture you want for an actual submission — the browser version was
built to run inside Claude's sandboxed artifact viewer only, where the API
call is proxied without exposing a key.
