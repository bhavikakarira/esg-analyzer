"""
AI ESG Scoring Engine
Uses Groq LLM instead of hardcoded rules.
"""

import json
import os
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------
# Prompt
# ---------------------------------------------------

SCORING_PROMPT = """
You are a senior ESG Rating Analyst.

Evaluate the ESG DISCLOSURE QUALITY of the company.

IMPORTANT

You are NOT evaluating the company's actual ESG performance.

Instead evaluate:

• completeness of reporting

• transparency

• measurable KPIs

• governance disclosure

• employee disclosure

• climate disclosure

• reporting maturity

A large multinational company with a comprehensive ESG report
should normally score between 75 and 95.

Do NOT give low scores simply because a few KPIs are missing.

Return ONLY valid JSON.

{
    "environmental": 0,
    "social": 0,
    "governance": 0,
    "overall": 0,
    "confidence": 0,
    "grade": "",
    "maturity": "",
    "strengths": [],
    "weaknesses": [],
    "recommendations": []
}

Extracted KPIs

<<DATA>>
"""
# ---------------------------------------------------
# JSON Cleaning
# ---------------------------------------------------

def clean_json(raw):

    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise Exception("No JSON found.")

    raw = raw[start:end+1]

    return json.loads(raw)


# ---------------------------------------------------
# LLM Scoring
# ---------------------------------------------------

def ai_score(kpis):

    prompt = SCORING_PROMPT.replace(
        "<<DATA>>",
        json.dumps(
            kpis,
            indent=2
        )
    )

    last_error = None

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                temperature=0,

                max_tokens=700,

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            return clean_json(
                response.choices[0].message.content
            )

        except Exception as e:

            last_error = e

            print(
                f"Retry {attempt+1}: {e}"
            )

            time.sleep(2)

    # Surface the real reason instead of a generic message so the caller
    # (and the fallback path) knows WHY the AI scoring failed.
    raise Exception(f"AI scoring failed after 3 attempts: {last_error}")
# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def _fallback_scores(kpis, reason="unknown"):
    """
    Fallback if AI scoring fails.
    Scores are based on disclosure coverage.

    IMPORTANT: this produces the SAME numbers for any report that has
    empty/near-empty KPI dicts (e.g. because extraction also failed).
    Callers must treat this as a degraded result, not a real score -
    it's flagged via fallback_used / fallback_reason below.
    """

    env = kpis.get("environmental", {})
    soc = kpis.get("social", {})
    gov = kpis.get("governance", {})

    def coverage(d):
        if not d:
            return 40

        valid = 0

        for v in d.values():
            if v not in [None, "", "null", "N/A"]:
                valid += 1

        score = min(40 + valid * 8, 95)

        return score

    e = coverage(env)
    s = coverage(soc)
    g = coverage(gov)

    overall = round(
        e * 0.4 +
        s * 0.3 +
        g * 0.3,
        1
    )

    return {

        "environmental": e,

        "social": s,

        "governance": g,

        "overall": overall,

        "confidence": 60,

        "grade": "B",

        "maturity": "Developing",

        "strengths": [],

        "weaknesses": [],

        "recommendations": [],

        "fallback_used": True,

        "fallback_reason": reason

    }


# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def get_all_scores(kpis):

    fallback_used = False
    fallback_reason = None

    try:
        scores = ai_score(kpis)
        scores["environmental"] = int(scores.get("environmental", 0))
        scores["social"] = int(scores.get("social", 0))
        scores["governance"] = int(scores.get("governance", 0))
        scores["overall"] = float(scores.get("overall", 0))
        scores["confidence"] = int(scores.get("confidence", 0))

    except Exception as e:
        print(e)
        fallback_used = True
        fallback_reason = str(e)
        scores = _fallback_scores(kpis, reason=fallback_reason)

    e = scores.get("environmental", 0)
    s = scores.get("social", 0)
    g = scores.get("governance", 0)

    # Smart weighting — if governance is very low (<30),
    # it likely means the report doesn't cover it,
    # not that the company is bad at governance.
    # Reduce its weight so it doesn't unfairly drag the score.
    if g < 30:
        overall = round(e * 0.55 + s * 0.35 + g * 0.10, 1)
    elif g < 50:
        overall = round(e * 0.45 + s * 0.35 + g * 0.20, 1)
    else:
        overall = round(e * 0.40 + s * 0.30 + g * 0.30, 1)

    scores["overall"] = overall

    # Grade based on recalculated overall
    if overall >= 85:
        grade = "A+"
    elif overall >= 75:
        grade = "A"
    elif overall >= 65:
        grade = "B+"
    elif overall >= 55:
        grade = "B"
    elif overall >= 45:
        grade = "C+"
    else:
        grade = "C"

    scores["grade"] = grade

    return {
        "E": e,
        "S": s,
        "G": g,
        "overall": overall,
        "confidence": scores.get("confidence", 0),
        "grade": grade,
        "maturity": scores.get("maturity", "N/A"),
        "coverage": {
            "environmental": e,
            "social": s,
            "governance": g
        },
        "missing": {
            "environmental": [],
            "social": [],
            "governance": []
        },
        "strengths": scores.get("strengths", []),
        "weaknesses": scores.get("weaknesses", []),
        "recommendations": scores.get("recommendations", []),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason
    }
