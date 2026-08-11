import json
import os
import time
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"

# -----------------------------------------------------
# Keywords used to locate relevant chunks
# -----------------------------------------------------

ENV_KEYWORDS = [
    "environment",
    "climate",
    "emission",
    "scope 1",
    "scope 2",
    "scope 3",
    "carbon",
    "renewable",
    "energy",
    "electricity",
    "water",
    "waste",
    "recycling",
    "net zero"
]

SOC_KEYWORDS = [
    "employee",
    "employees",
    "people",
    "our people",
    "talent",
    "workforce",
    "learning",
    "training",
    "skill",
    "upskilling",
    "diversity",
    "inclusion",
    "belonging",
    "women",
    "gender",
    "community",
    "volunteer",
    "wellbeing",
    "health",
    "safety",
    "injury",
    "human capital",
    "culture"
]

GOV_KEYWORDS = [
    "board",
    "board of directors",
    "director",
    "governance",
    "corporate governance",
    "audit",
    "audit committee",
    "committee",
    "ethics",
    "code of conduct",
    "compliance",
    "risk management",
    "cybersecurity",
    "privacy",
    "anti corruption",
    "whistleblower",
    "esg committee"
]
ENV_PROMPT = """
Extract ONLY Environmental KPIs.

Return ONLY JSON.

{
"co2_emissions":null,
"scope1_emissions":null,
"scope2_emissions":null,
"scope3_emissions":null,
"renewable_energy_percent":null,
"energy_consumption":null,
"water_consumption":null,
"waste_generated":null,
"net_zero_target":null,
"science_based_targets":null
}

TEXT

<<TEXT>>
"""

SOC_PROMPT = """
Extract ONLY Social KPIs.

Return ONLY JSON.

{
"total_employees":null,
"women_in_workforce_percent":null,
"women_in_leadership_percent":null,
"employee_training_hours":null,
"injury_rate":null,
"community_investment":null,
"diversity_programs":null
}

TEXT

<<TEXT>>
"""

GOV_PROMPT = """
Extract ONLY Governance KPIs.

Return ONLY JSON.

{
"board_size":null,
"women_on_board_percent":null,
"independent_directors_percent":null,
"audit_committee":null,
"board_diversity":null,
"ethics_training":null,
"anti_corruption_cases":null,
"cybersecurity":null,
"esg_policy_disclosed":null
}

TEXT

<<TEXT>>
"""
def score_chunk(chunk, keywords):

    text = chunk.lower()

    score = 0

    for word in keywords:

        if word in text:
            score += 1

    return score


def best_chunks(chunks, keywords, top_n=12):

    ranked = sorted(
        chunks,
        key=lambda x: score_chunk(x, keywords),
        reverse=True
    )

    return ranked[:top_n]


def clean_json(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        raise Exception("No JSON returned")

    return json.loads(text[start:end+1])
# -----------------------------------------------------
# LLM Call
# -----------------------------------------------------

def query_llm(prompt, chunk):

    final_prompt = prompt.replace("<<TEXT>>", chunk)

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                temperature=0,

                max_tokens=500,

                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ]

            )

            raw = response.choices[0].message.content

            return clean_json(raw)

        except Exception as e:

            print(f"Retry {attempt+1}: {e}")

            time.sleep(2)

    return {}


# -----------------------------------------------------
# Merge Results
# -----------------------------------------------------

def merge_dict(master, new):

    for key, value in new.items():

        if value in [None, "", "null", "N/A", "Unknown"]:
            continue

        if key not in master:

            master[key] = value

            continue

        if master[key] in [None, "", "null"]:

            master[key] = value

    return master


# -----------------------------------------------------
# Environmental Extraction
# -----------------------------------------------------

def extract_environmental(chunks):

    result = {}

    env_chunks = best_chunks(
        chunks,
        ENV_KEYWORDS,
        top_n=12
    )

    print(f"\nEnvironmental Chunks: {len(env_chunks)}")

    for chunk in env_chunks:

        try:

            data = query_llm(
                ENV_PROMPT,
                chunk
            )

            result = merge_dict(
                result,
                data
            )

        except Exception as e:

            print(e)

    return result


# -----------------------------------------------------
# Social Extraction
# -----------------------------------------------------

def extract_social(chunks):

    result = {}

    soc_chunks = best_chunks(
        chunks,
        SOC_KEYWORDS,
        top_n=12
    )

    print(f"\nSocial Chunks: {len(soc_chunks)}")

    for chunk in soc_chunks:

        try:

            data = query_llm(
                SOC_PROMPT,
                chunk
            )

            result = merge_dict(
                result,
                data
            )

        except Exception as e:

            print(e)

    return result


# -----------------------------------------------------
# Governance Extraction
# -----------------------------------------------------

def extract_governance(chunks):

    result = {}

    gov_chunks = best_chunks(
        chunks,
        GOV_KEYWORDS,
        top_n=12
    )

    print(f"\nGovernance Chunks: {len(gov_chunks)}")

    for chunk in gov_chunks:

        try:

            data = query_llm(
                GOV_PROMPT,
                chunk
            )

            result = merge_dict(
                result,
                data
            )

        except Exception as e:

            print(e)

    return result


# -----------------------------------------------------
# Main Function
# -----------------------------------------------------

def extract_kpis(chunks):

    print("\nStarting ESG Extraction...\n")

    environmental = extract_environmental(chunks)

    social = extract_social(chunks)

    governance = extract_governance(chunks)

    merged = {

        "environmental": environmental,

        "social": social,

        "governance": governance

    }

    print("\n============================")
    print("FINAL ESG KPIs")
    print("============================")

    print(json.dumps(
        merged,
        indent=4
    ))

    return merged
