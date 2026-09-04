"""
app.py — Hasha AI Teacher backend.

Mirrors the exact teaching loop used in the browser prototype, but running
server-side so:
  1. The Anthropic API key never touches the client.
  2. RAG retrieval runs with real TF-IDF instead of the browser's naive
     keyword-count fallback.
  3. Learner profiles persist properly (a JSON file here; swap for a real
     DB in production).

Run:
    pip install -r requirements.txt
    cp .env.example .env   # add your ANTHROPIC_API_KEY
    python app.py

Then point any frontend (the included HTML prototype, a React app, curl,
Postman, etc.) at http://localhost:5000/api/...
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from rag import chunk_material, TfidfRetriever
from claude_client import call_json

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow the browser prototype (or any frontend) to call this API

PROFILE_PATH = Path(__file__).parent / "profiles.json"

LANG_HINT = {
    "English": "Respond only in English.",
    "Hindi": "Respond only in Hindi, Devanagari script.",
    "Hinglish": "Respond only in Hinglish — mix Hindi and English naturally, in Latin script.",
    "Tamil": "Respond only in Tamil, Tamil script.",
    "Tanglish": "Respond only in Tanglish — mix Tamil and English naturally, in Latin script.",
}


# ---------------------------------------------------------------------------
# Learner profile persistence (simple JSON file; swap for a real DB later)
# ---------------------------------------------------------------------------

def _load_profiles() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text())
    return {}


def _save_profiles(data: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(data, indent=2))


def _default_profile() -> dict:
    return {"strong": [], "weak": [], "history": []}


@app.route("/api/profile/<student_id>", methods=["GET"])
def get_profile(student_id: str):
    profiles = _load_profiles()
    return jsonify(profiles.get(student_id, _default_profile()))


@app.route("/api/profile/<student_id>", methods=["POST"])
def update_profile(student_id: str):
    profiles = _load_profiles()
    profiles[student_id] = request.json
    _save_profiles(profiles)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# 1. Lesson plan
# ---------------------------------------------------------------------------

@app.route("/api/plan", methods=["POST"])
def generate_plan():
    body = request.json
    topic = body["topic"]
    level = body.get("level", "Beginner")
    time_budget = body.get("time", "20 minutes")
    language = body.get("language", "English")
    material = body.get("material", "")

    grounding_note = (
        f'The student uploaded material. Base the concept list on this material:\n"""{material[:3000]}"""'
        if material.strip()
        else "No material uploaded. Build the concept list from your own knowledge of the topic."
    )

    system = "You are a curriculum planner for an AI teacher. Output ONLY valid JSON, no prose, no markdown fences."
    user = f"""Topic: "{topic}"
Learner level: {level}
Available time: {time_budget}
Teaching language: {language}
{grounding_note}

Return JSON: {{"concepts":[{{"title":"...","why":"one line on why this comes at this point"}}]}}
Number of concepts should fit the time budget (5 min -> 2 concepts, 20 min -> 3-4, 60 min -> 5-6, 7 days -> 6-8 as a revision path). Order concepts from foundational to advanced, respecting prerequisites."""

    result = call_json(system, user, max_tokens=700)
    if not result or "concepts" not in result:
        return jsonify({"error": "Could not generate a lesson plan. Try again."}), 502
    return jsonify(result)


# ---------------------------------------------------------------------------
# 2. Concept teaching (explanation + visual spec + checkpoint question)
# ---------------------------------------------------------------------------

@app.route("/api/concept", methods=["POST"])
def generate_concept():
    body = request.json
    topic = body["topic"]
    concept_title = body["concept_title"]
    level = body.get("level", "Beginner")
    language = body.get("language", "English")
    material = body.get("material", "")
    alt_analogy = body.get("alt_analogy", False)

    chunks = chunk_material(material) if material.strip() else []
    retriever = TfidfRetriever(chunks)
    grounded_chunks = retriever.retrieve(concept_title, top_k=2)

    grounding_block = (
        f'Ground your explanation in this source material where relevant:\n"""{"---".join(grounded_chunks)}"""'
        if grounded_chunks
        else "No source material; use your own accurate knowledge."
    )
    lang_hint = LANG_HINT.get(language, "Respond only in English.")
    alt_hint = (
        "The student struggled last time — use a DIFFERENT analogy/approach than a typical explanation."
        if alt_analogy
        else ""
    )

    system = (
        f"You are a warm, human-like AI teacher speaking directly to a {level} learner. "
        f"{lang_hint} Output ONLY valid JSON, no prose outside JSON."
    )
    user = f"""Teach this concept: "{concept_title}" (part of a lesson on "{topic}").
{grounding_block}
{alt_hint}

Return JSON exactly like:
{{
 "explanation": "2-4 short paragraphs, {level}-appropriate, spoken teaching tone",
 "visual": {{"type":"equation|diagram_steps|timeline|code|graph_bar|bullets", "data": {{ ... fields matching the type ... }}}},
 "question": {{
   "prompt":"a checkpoint question testing this concept",
   "format":"mcq or short_answer",
   "options":["only if mcq, 3-4 options"],
   "correctAnswer":"the correct option text or ideal short answer"
 }}
}}
Pick the visual type that best fits the subject (equation for math/physics formulas, diagram_steps for processes, timeline for historical/sequential events, code for programming, graph_bar for comparative data, bullets otherwise)."""

    result = call_json(system, user, max_tokens=1000)
    if not result:
        return jsonify({"error": "Could not generate this concept. Try again."}), 502

    result["groundedFrom"] = len(grounded_chunks) > 0
    return jsonify(result)


# ---------------------------------------------------------------------------
# 3. Answer evaluation + misconception detection
# ---------------------------------------------------------------------------

@app.route("/api/evaluate", methods=["POST"])
def evaluate_answer():
    body = request.json
    question_prompt = body["question_prompt"]
    correct_answer = body["correct_answer"]
    student_answer = body["student_answer"]
    language = body.get("language", "English")

    system = "You are evaluating a student's answer as an adaptive AI teacher. Output ONLY valid JSON."
    user = f"""Question: "{question_prompt}"
Correct answer: "{correct_answer}"
Student answered: "{student_answer}"

Return JSON:
{{"correct": true/false, "misconception": "if wrong, name the specific misunderstanding in one sentence, else empty string", "feedback": "1-3 sentence constructive feedback in {language}", "nextAction": "continue or reteach"}}
Mark "reteach" only if the answer reveals a real conceptual gap, not a minor wording issue."""

    result = call_json(system, user, max_tokens=500)
    if not result:
        return jsonify({"error": "Could not evaluate this answer. Try again."}), 502
    return jsonify(result)


# ---------------------------------------------------------------------------
# 4. Final quiz
# ---------------------------------------------------------------------------

@app.route("/api/quiz", methods=["POST"])
def generate_quiz():
    body = request.json
    topic = body["topic"]
    concept_titles = body["concept_titles"]  # list[str]
    level = body.get("level", "Beginner")
    language = body.get("language", "English")

    system = "You write a short final quiz. Output ONLY valid JSON."
    user = f"""Lesson topic: "{topic}". Concepts covered: {", ".join(concept_titles)}. Level: {level}. Language: {language}.
Return JSON: {{"questions":[{{"prompt":"...","options":["a","b","c","d"],"correctAnswer":"exact matching option text"}}]}} with exactly 4 MCQ questions, one per concept where possible, in {language}."""

    result = call_json(system, user, max_tokens=900)
    if not result or "questions" not in result:
        return jsonify({"error": "Could not generate the quiz. Try again."}), 502
    return jsonify(result)


# ---------------------------------------------------------------------------
# 5. Learning report
# ---------------------------------------------------------------------------

@app.route("/api/report", methods=["POST"])
def generate_report():
    body = request.json
    topic = body["topic"]
    score = body["score"]
    missed = body.get("missed_questions", [])
    correct = body.get("correct_questions", [])
    profile_weak = body.get("profile_weak", [])
    language = body.get("language", "English")

    system = "You write a short encouraging learning report. Output ONLY valid JSON."
    user = f"""Topic: "{topic}". Score: {score}%. Missed questions: {json.dumps(missed)}. Correct questions: {json.dumps(correct)}. Profile weak areas so far: {json.dumps(profile_weak)}.
Return JSON: {{"summary":"2 sentence encouraging summary in {language}", "strongAreas":["short concept names"], "weakAreas":["short concept names"], "recommendation":"one specific next step in {language}"}}"""

    result = call_json(system, user, max_tokens=500)
    if not result:
        return jsonify({"error": "Could not generate the report. Try again."}), 502
    result["score"] = score
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
