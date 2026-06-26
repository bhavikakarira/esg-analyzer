import json
import os
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

GREEN_KEYWORDS = [
    "sustainability", "sustainable", "climate", "environment", "esg",
    "carbon", "net zero", "renewable", "green", "future", "impact",
    "responsible", "planet", "nature", "biodiversity", "commitment",
    "ambition", "decarbonization", "low carbon", "emissions"
]

GREEN_PROMPT = """
You are a certified ESG auditor.

Your task is NOT to accuse the company.

Instead identify sustainability claims that could benefit from stronger evidence.

A claim should ONLY be flagged if one or more of the following is true:
• vague wording with no measurable KPI
• no timeline
• no supporting evidence
• no independent verification

Ignore claims that are already supported by numbers, targets or third-party assurance.

Return ONLY a JSON list. Maximum 5 items. If no concerns return [].

[
  {
    "claim": "short quote under 100 chars",
    "evidence_present": "YES or NO",
    "timeline_present": "YES or NO",
    "numerical_data": "YES or NO",
    "third_party_verification": "YES or NO",
    "reason": "one sentence explanation",
    "risk": "LOW or MEDIUM or HIGH"
  }
]

TEXT

<<TEXT>>
"""


def score_chunk(chunk, keywords):
    text = chunk.lower()
    return sum(1 for word in keywords if word in text)


def best_chunks(chunks, keywords, top_n=6):
    ranked = sorted(chunks, key=lambda c: score_chunk(c, keywords), reverse=True)
    return ranked[:top_n]


def clean_json(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(raw[start:end+1])


def analyze_chunk(chunk):
    prompt = GREEN_PROMPT.replace("<<TEXT>>", chunk)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            return clean_json(response.choices[0].message.content)
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(3)
    return []


def detect_greenwashing(chunks):
    all_flags = []
    selected = best_chunks(chunks, GREEN_KEYWORDS, top_n=6)

    print(f"\nAnalyzing {len(selected)} sustainability chunks...\n")

    for i, chunk in enumerate(selected):
        print(f"Chunk {i+1}/{len(selected)}")
        try:
            flags = analyze_chunk(chunk)
            if isinstance(flags, list):
                all_flags.extend(flags)
        except Exception as e:
            print(e)
        time.sleep(1)

    # Deduplicate
    unique = []
    seen = set()
    for flag in all_flags:
        claim = " ".join(flag.get("claim", "").lower().split())
        if claim and claim not in seen:
            seen.add(claim)
            unique.append(flag)

    return unique


def calculate_risk_score(flags):
    """
    Fixed scoring — caps at 100, doesn't inflate on large reports.
    Normalises by number of flags so a report with 30 medium flags
    doesn't automatically hit 100.
    """
    if not flags:
        return 5

    raw = 0
    for flag in flags:
        risk = flag.get("risk", "LOW").upper()
        evidence = flag.get("evidence_present", "NO").upper()
        timeline = flag.get("timeline_present", "NO").upper()
        numbers = flag.get("numerical_data", "NO").upper()
        verification = flag.get("third_party_verification", "NO").upper()

        if risk == "HIGH":
            raw += 15
        elif risk == "MEDIUM":
            raw += 7
        else:
            raw += 2

        # Reduce when evidence exists
        if evidence == "YES":
            raw -= 2
        if timeline == "YES":
            raw -= 2
        if numbers == "YES":
            raw -= 2
        if verification == "YES":
            raw -= 4

    # Normalise: divide by expected max so score stays meaningful
    # A report with 10 HIGH flags = roughly 150 raw → 100 capped
    # A report with 18 MEDIUM flags = roughly 126 raw → still reasonable
    normalised = min(int((raw / max(len(flags), 1)) * 10), 100)
    normalised = max(0, normalised)

    return normalised


def summarize_greenwashing(flags):
    total = len(flags)
    high = sum(1 for f in flags if f.get("risk", "").upper() == "HIGH")
    medium = sum(1 for f in flags if f.get("risk", "").upper() == "MEDIUM")
    low = sum(1 for f in flags if f.get("risk", "").upper() == "LOW")

    if total == 0:
        verdict = "No significant greenwashing indicators detected. Claims appear well-supported by evidence."
    elif high >= 5:
        verdict = "High likelihood of unsupported sustainability claims. Several statements lack measurable evidence."
    elif high >= 2:
        verdict = "Moderate greenwashing risk. Some claims would benefit from stronger evidence."
    elif medium >= 3:
        verdict = "Generally credible reporting with a few claims that could use stronger supporting evidence."
    else:
        verdict = "Most sustainability disclosures appear transparent and evidence-backed."

    return {"total_flags": total, "high": high, "medium": medium, "low": low, "verdict": verdict}