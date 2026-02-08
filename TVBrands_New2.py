# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 23:59:33 2026

@author: zubin
"""

# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, re, requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from openai import OpenAI
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

# ================== CONFIG ==================
load_dotenv()

SERP_API_KEY = os.getenv("SERPAPI_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
sentiment_analyzer = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="TV Brand Visibility Analyzer – India",
    page_icon="📺",
    layout="wide"
)

MAX_SERP_QUERIES = 20
PERPLEXITY_DEEP_CACHE = {}


API_COUNTERS = {
    "serpapi": 0,
    "chatgpt": 0,
    "perplexity": 0
}

HISTORY_FILE = "tv_brand_visibility_history.csv"

CONFIDENCE_ALERT_THRESHOLD = 40

MAX_QUERIES_BY_MODE = {
    "Fast Preview": 6,
    "Deep Research": 10
}


# ================== BRAND ALIASES ==================
BRAND_ALIASES = {
    "Lumio": ["lumio", "lumio tv", "vision 7", "vision 9", "boss processor"],
    "Samsung": ["samsung", "qled", "crystal 4k"],
    "LG": ["lg", "oled", "nanocell"],
    "Sony": ["sony", "bravia"],
    "Mi/Xiaomi": ["mi tv", "xiaomi", "redmi tv"],
    "OnePlus": ["oneplus tv", "y1", "u1"],
    "TCL": ["tcl", "iffalcon"],
    "Hisense": ["hisense", "u7k"],
    "VU": ["vu tv"],
    "Thomson": ["thomson tv"]
}

EXPANDED_FEATURES = [
    "Picture Quality", "Sound Quality", "Value for Money",
    "Boot-up Speed", "UI Fluidity", "On-Device AI", 
    "Gaming Tech (VRR/144Hz)", "Service Reach (India)"
]

INTENT_WEIGHTS = {
    "complaint": 1.5,
    "comparison": 1.2,
    "buying": 1.0,
    "generic": 0.8
}

def classify_query_intent(query):
    q = query.lower()
    if any(k in q for k in ["complaint", "issue", "problem", "worst", "bad", "fault"]):
        return "complaint"
    if any(k in q for k in ["vs", "compare", "comparison"]):
        return "comparison"
    if any(k in q for k in ["best", "top", "buy", "recommend"]):
        return "buying"
    return "generic"


def classify_sentiment_bucket(text, weight=1.0):
    score = sentiment_score(text) * weight
    if score <= -0.3:
        return "complaint"
    if score >= 0.3:
        return "praise"
    return "neutral"

# ================== SENTIMENT ==================
def sentiment_score(text):
    return sentiment_analyzer.polarity_scores(text)["compound"]

# ================== TRENDS ==================
@st.cache_data(ttl=3600)
def fetch_trending_queries(base_queries):
    rows = []
    for q in base_queries:
        rows.append({"query": q.lower(), "volume": 100})
    return rows

# ================== SERP SEARCH ==================
@st.cache_data(ttl=86400)
def serp_search(query):
    params = {
        "engine": "google",
        "q": query,
        "gl": "in",
        "hl": "en",
        "num": 5,
        "api_key": SERP_API_KEY
    }
    r = requests.get("https://serpapi.com/search", params=params)
    API_COUNTERS["serpapi"] += 1
    if not r.ok:
        return []

    data = r.json().get("organic_results", [])
    return [{
        "text": f"{d.get('title','')} {d.get('snippet','')}",
        "link": d.get("link","")
    } for d in data]

# ================== CHATGPT ==================
# =============================================================================
# def extract_chatgpt_questions(keyword, serp_data):
#     context = "\n".join([d["text"] for d in serp_data])
#     prompt = f"""
#     Based on these Google Search results for '{keyword}' in India, 
#     identify 3 specific, real-world questions an Indian consumer is likely asking.
#     Focus on specific pain points (e.g., 'Is 55 inch too big for a 10ft room?' or 'Does Lumio have service centers in Bangalore?').
#     
#     Context:
#     {context}
#     
#     Return only the questions as a bulleted list.
#     """
#     API_COUNTERS["chatgpt"] += 1
#     r = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7 # Higher temperature for more natural/varied questions
#     )
#     return r.choices[0].message.content
# =============================================================================
# ================== PERPLEXITY ==================
# =============================================================================
# def fetch_perplexity_consumer_queries(keyword):
#     payload = {
#         "model": "sonar",
#         "messages": [
#             {
#                 "role": "user", 
#                 "content": f"Search the web for the most common questions Indian buyers are asking about {keyword} in 2026. "
#                            f"Provide 5 'Long-tail' questions that show high buying intent. "
#                            f"Include questions about specific brands like Lumio, Samsung, or Sony where relevant."
#             }
#         ],
#         "temperature": 0.5,
#         "return_citations": True
#     }
#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     r = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
#     API_COUNTERS["perplexity"] += 1
#     data = r.json()
#     return data["choices"][0]["message"]["content"]
# =============================================================================
    

def chatgpt_question_generator(keyword, serp_data):
    context = "\n".join([d["text"] for d in serp_data])
    links = [d["link"] for d in serp_data if d["link"]]
    
    prompt = f"Based on these results for '{keyword}', what are 3 specific questions Indian buyers are asking? Return only the questions."
    
    API_COUNTERS["chatgpt"] += 1
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\n{prompt}"}]
    )
    return r.choices[0].message.content, links

# def perplexity_question_miner(keyword):
#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "sonar",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"List the most common real-world questions Indian consumers ask online when researching about {keyword}?"
#             }
#         ],
#         "return_citations": True
#     }

#     try:
#         r = requests.post(
#             "https://api.perplexity.ai/chat/completions",
#             headers=headers,
#             json=payload,
#             timeout=40
#         )

#         # ❗ HARD GUARD 1 — empty or non-JSON response
#         if not r.text or not r.text.strip().startswith("{"):
#             return "Perplexity questions unavailable.", []

#         data = r.json()

#         questions = data["choices"][0]["message"]["content"]

#         sources = [
#             c["url"] for c in data.get("citations", [])
#             if isinstance(c, dict) and "url" in c
#         ]

#         return questions, sources

#     except Exception as e:
#         # ❗ HARD GUARD 2 — never crash Streamlit
#         return f"Perplexity error: {str(e)}", []



def perplexity_answer(keyword):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "user",
                "content": f"Answer this as an Indian TV buying guide in 2026: {keyword}"
            }
        ],
        "return_citations": True
    }

    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=40
        )

        # ❗ HARD GUARD 1 — empty or non-JSON response
        if not r.text or not r.text.strip().startswith("{"):
            return "Perplexity response unavailable.", []

        data = r.json()

        answer = data["choices"][0]["message"]["content"]
        sources = [
            c["url"] for c in data.get("citations", [])
            if isinstance(c, dict) and "url" in c
        ]

        return answer, sources

    except Exception as e:
        # ❗ HARD GUARD 2 — never crash Streamlit
        return f"Perplexity error: {str(e)}", []


def chatgpt_answer_question(question, context=""):
    prompt = f"""
Answer the following question for an Indian TV buyer in 2026.
Be practical and concise.

Question:
{question}

Context (if any):
{context}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# ================== BRAND EXTRACTION ==================
def extract_mentions(texts):
    mentions = defaultdict(lambda: {"count": 0, "sentences": []})
    for t in texts:
        tl = t.lower()
        for brand, aliases in BRAND_ALIASES.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", tl):
                    mentions[brand]["count"] += 1
                    sentences = re.split(r'[.!?]', t)
                    for s in sentences:
                        if re.search(rf"\b{re.escape(a)}\b", s.lower()):
                            mentions[brand]["sentences"].append(s.strip())
    return mentions


def generate_expanded_queries(segment, user_seed_keywords):
    """
    Uses AI to expand 1-2 seed keywords into a 10-item search strategy
    tailored to the 2026 Indian TV market.
    """
    segment_context = {
        "Budget": "focus on price sensitivity, durability, and essential smart features under ₹30k",
        "Premium": "focus on OLED/Mini-LED, AI upscaling, 144Hz gaming, and design aesthetic",
        "All": "a mix of market leaders, trending tech, and general 'best of' lists"
    }

    prompt = f"""
    Act as a Senior SEO Strategist for the Indian Consumer Electronics market in 2026.
    Expand these seed keywords: {user_seed_keywords}
    
    Target Segment: {segment} ({segment_context.get(segment)})
    Generate 10 unique, high-intent search queries. Include:
    1. Comparison queries (e.g., 'Brand A vs Brand B 2026')
    2. Performance queries (e.g., 'fastest boot up smart tv India')
    3. Sentiment queries (e.g., 'most reliable tv brand reviews India 2026')
    
    Return ONLY the queries as a Python list of strings.
    """
    
    API_COUNTERS["chatgpt"] += 1
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 # Higher temperature for more creative search variations
    )
    
    # Simple parsing to ensure we get a clean list
    try:
        expanded_list = eval(response.choices[0].message.content.strip())
        return expanded_list[:10] # Hard limit to 10 to manage API costs
    except:
        # Fallback if AI formatting fails
        return user_seed_keywords * 5
    
def generate_perplexity_prompts(brand, segment):
    """
    Generates high-context research prompts for Perplexity.
    Unlike Google queries, these are full sentences designed to trigger 
    deep research into brand sentiment and technical gaps.
    """
    prompts = [
        f"Analyze the 2026 market reputation of {brand} TVs in India regarding boot-up speed and OS lag compared to top-tier brands.",
        f"What are the most common technical complaints for {brand} smart TVs on Indian consumer forums in the last 6 months?",
        f"Compare {brand}'s service network reliability in Tier 2 Indian cities against Samsung and Xiaomi.",
        f"Does {brand} use dedicated NPUs for on-device AI upscaling in their 2026 models? Cite recent reviews."
    ]
    return prompts

def run_perplexity_deep_research(brand, segment):
    prompts = generate_perplexity_prompts(brand, segment)
    combined_answer = ""

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    for p in prompts:
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": p}],
            "return_citations": True
        }
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        API_COUNTERS["perplexity"] += 1
        try:
            data = r.json()
            combined_answer += " " + data["choices"][0]["message"]["content"]
        except Exception:
            continue

    return combined_answer.strip()    

@st.cache_data(ttl=86400)
def cached_perplexity_research(brand, segment):
    return run_perplexity_deep_research(brand, segment)

# ================== LLM-POWERED EXTRACTION OF FEATURES ==================
@st.cache_data(ttl=86400)
def extract_competitive_insights(brand, sentences):
    """Refined to specifically target Boot-up Speed and UI performance."""
    if not sentences:
        return {f: "N/A" for f in EXPANDED_FEATURES}
    
    clean_sentences = [
        s if isinstance(s, str) else s[0]
        for s in sentences
    ]
    context = " ".join(clean_sentences)[:4000]
    prompt = f"""
    Analyze the brand '{brand}' for the Indian market in 2026.
    Rate these features: {', '.join(EXPANDED_FEATURES)}.
    
    SPECIAL FOCUS: 
    - 'Boot-up Speed': Is it 'Instant' (under 3s), 'Fast' (3-10s), or 'Slow' (>15s)?
    - 'UI Fluidity': Does the menu lag or stutter?
    
    Context: {context}
    Return ONLY a comma-separated list of results (e.g., Excellent, Fast, Good...).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        scores = response.choices[0].message.content.strip().split(",")
        return dict(zip(EXPANDED_FEATURES, [s.strip() for s in scores]))
    except:
        return {f: "Error" for f in EXPANDED_FEATURES}
    
@st.cache_data(ttl=86400)
def perplexity_query_answer(query):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [{
            "role": "user",
            "content": f"""
Answer this as an Indian TV buying expert in 2026.

Question:
{query}

Focus on brand-wise strengths, weaknesses, and reliability.
"""
        }],
        "return_citations": True
    }

    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
        timeout=40
    )

    if not r.ok:
        return "Perplexity unavailable.", []

    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    sources = [
        c["url"] for c in data.get("citations", [])
        if isinstance(c, dict) and "url" in c
    ]

    return answer, sources
    
def extract_perplex_competitive_insights(brand, segment):
    """
    Uses Perplexity ONLY to extract structured competitive insights
    in strict JSON format.
    """

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a market research analyst.

Analyze the TV brand "{brand}" in India (2026).

Return STRICT JSON with EXACTLY these keys:
{EXPANDED_FEATURES}

Rules:
- Values must be one of: "Excellent", "Good", "Average", "Poor", "Unknown"
- Do NOT add extra keys
- Do NOT add explanations
- Do NOT return text outside JSON
- If data is insufficient, use "Unknown"

JSON ONLY.
"""

    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "return_citations": False
    }

    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=40
        )

        data = r.json()
        raw = data["choices"][0]["message"]["content"]

        parsed = json.loads(raw)

        # 🔒 Hard validation
        if set(parsed.keys()) != set(EXPANDED_FEATURES):
            raise ValueError("Schema mismatch")

        return parsed

    except Exception:
        return {f: "N/A" for f in EXPANDED_FEATURES}
 


def generate_lumio_recommendations(diagnostics):
    prompt = f"""
You are a senior strategy consultant for an Indian consumer electronics brand (2026).

Brand: Lumio

Observed signals:

SERP + ChatGPT:
- Visibility status: { "Not visible at all" if diagnostics["serp_visibility_status"]=="not_visible" else "Visible" }
- Visibility gap vs competitors: {diagnostics["serp_visibility_gap"]}
- Sentiment status: { "Sentiment not found due to low visibility" if diagnostics["serp_sentiment_status"]=="not_available" else "Available" }
- Sentiment gap vs competitors: {diagnostics["serp_sentiment_gap"]}

Perplexity:
- Visibility status: { "Not visible at all" if diagnostics["pplx_visibility_status"]=="not_visible" else "Visible" }
- Visibility gap vs competitors: {diagnostics["pplx_visibility_gap"]}
- Sentiment status: { "Sentiment not found due to low visibility" if diagnostics["pplx_sentiment_status"]=="not_available" else "Available" }
- Sentiment gap vs competitors: {diagnostics["pplx_sentiment_gap"]}


- Top competing brands: {", ".join(diagnostics["top_competitors"])}

User sentiment excerpts:
{chr(10).join(diagnostics["lumio_sentences"][:15])}

TASK:
Provide concrete, non-generic recommendations for Lumio to improve:
1. Search visibility & discovery
2. Product & UX perception
3. Messaging & positioning
4. Trust, service & reliability

RULES:
- Use bullet points
- Be India-specific
- Reference issues implied by sentiment (e.g. lag, service, pricing)
- Avoid fluff or generic marketing advice

Return in this format:

SEO & DISCOVERY:
- ...

PRODUCT & UX:
- ...

MESSAGING:
- ...

TRUST & SERVICE:
- ...
"""

    API_COUNTERS["chatgpt"] += 1
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

def safe_avg_sentiment(sentences, cap=20):
    if not sentences:
        return 0

    total, weight_sum = 0, 0

    for item in sentences[:cap]:
        if isinstance(item, tuple):
            s, w = item
        else:
            s, w = item, 1.0  # default weight for SERP / ChatGPT

        total += sentiment_score(s) * w
        weight_sum += w

    return round(total / weight_sum, 3) if weight_sum else 0

def compute_sentiment_index(sentences, cap=30):
    """
    Pure sentiment polarity index (-1 to +1)
    Uses intent weight ONLY as frequency weight, not polarity scaler
    """
    if not sentences:
        return 0.0

    total = 0.0
    weight_sum = 0.0

    for item in sentences[:cap]:
        if isinstance(item, tuple):
            s, w = item
        else:
            s, w = item, 1.0

        total += sentiment_score(s)
        weight_sum += w

    return round(total / weight_sum, 3) if weight_sum else 0.0


def compute_risk_index(complaints, praise):
    """
    Risk = complaint density
    Range: 0 (no risk) → 1 (high risk)
    """
    total = complaints + praise
    if total == 0:
        return 0.0

    return round(complaints / total, 3)


def extract_perplex_competitive_insights_from_text(brand, context_text):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a market research analyst.

Using ONLY the information below, analyze the TV brand "{brand}" in India (2026).

Context:
{context_text[:4000]}

Return STRICT JSON with EXACTLY these keys:
{EXPANDED_FEATURES}

Rules:
- Values must be one of: "Excellent", "Good", "Average", "Poor", "Unknown"
- Do NOT add extra keys
- Do NOT add explanations
- Do NOT return text outside JSON
- Base ratings ONLY on the provided context
"""

    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "return_citations": False
    }

    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=40
        )

        data = r.json()
        parsed = json.loads(data["choices"][0]["message"]["content"])

        if set(parsed.keys()) != set(EXPANDED_FEATURES):
            raise ValueError("Schema mismatch")

        return parsed

    except Exception:
        return {f: "N/A" for f in EXPANDED_FEATURES}


# ================== UPDATED ANALYSIS LOOP ==================


def run_analysis(queries):
    # Initialize the storage dictionaries that were missing
    progress = st.progress(0)
    effective_queries = queries[:MAX_QUERIES_BY_MODE[analysis_mode]]
    total = len(effective_queries)
    serp_results = defaultdict(lambda: {"mentions": 0, "sentences": []})
    pplx_results = defaultdict(
        lambda: {
            "mentions": 0,
            "sentences": [],
            "complaints": 0,
            "praise": 0
        }
    )

    answers = []
    all_pplx_brands = set()
    

    for i, q in enumerate(effective_queries):
        # 1. Get SERP Data
        serp_data = serp_search(q["query"])
        
        # 2. Extract Brand Mentions from SERP
        serp_mentions = extract_mentions([d["text"] for d in serp_data])
        for brand, data in serp_mentions.items():
            serp_results[brand]["mentions"] += data["count"]
            query_intent = classify_query_intent(q["query"])
            intent_weight = INTENT_WEIGHTS.get(query_intent, 1.0)
            
            for s in data["sentences"]:
                serp_results[brand]["sentences"].append((s, intent_weight))


        # 3. Get Questions & Sources
        chatgpt_q, gsrc = chatgpt_question_generator(q["query"], serp_data)
        chatgpt_mentions = extract_mentions([chatgpt_q])
        primary_question = chatgpt_q.strip().split("\n")[0]
        if analysis_mode != "Fast Preview":
            chatgpt_answer_text = chatgpt_answer_question(
                primary_question,
                context=" ".join([d["text"] for d in serp_data[:3]])
            )
        else:
            chatgpt_answer_text = "Skipped in Fast Preview"
      
                    
        if PERPLEXITY_API_KEY:
            # ✅ Always run ONE Perplexity call (even in Fast Preview)
            pplx_answer, pplx_answer_sources = perplexity_query_answer(q["query"])
            pplx_questions = q["query"]
        
            query_intent = classify_query_intent(q["query"])
            intent_weight = INTENT_WEIGHTS.get(query_intent, 1.0)
        
            pplx_question_sources = []
        else:
            pplx_answer = "Perplexity disabled."
            pplx_answer_sources = []
            pplx_questions = "Perplexity disabled."
            pplx_question_sources = []

    
        # 4. Simple logic to find "Top Brands" for this specific query
        # We sort brands found in this specific search by mention count
        sorted_serp = sorted(serp_mentions.items(), key=lambda x: x[1]["count"], reverse=True)
        top_serp = [b[0] for b in sorted_serp[:3]]
              
        if not PERPLEXITY_API_KEY:
            pplx_answer = "Perplexity disabled."
            pplx_questions = "Perplexity disabled."
            pplx_answer_sources = []
            pplx_question_sources = []
            pplx_mentions = {}
        else:
            pplx_mentions = extract_mentions([pplx_answer])

            # 🔧 Fallback: if no brands detected, assume answer applies to top SERP brands
            if not pplx_mentions:
                pplx_mentions = {}
            

        # 1️⃣ Aggregate basic Perplexity signals (mentions-based)
        for brand, data in pplx_mentions.items():
            pplx_results[brand]["mentions"] += data["count"]
            for s in data["sentences"]:
                pplx_results[brand]["sentences"].append((s, intent_weight))
                bucket = classify_sentiment_bucket(s, intent_weight)
                if bucket == "complaint":
                    pplx_results[brand]["complaints"] += 1
                elif bucket == "praise":
                    pplx_results[brand]["praise"] += 1
        

        # 2️⃣ Define top Perplexity brands for THIS query
        sorted_pplx = sorted(
            pplx_mentions.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        top_pplx = [b[0] for b in sorted_pplx[:3]]
        all_pplx_brands.update(top_pplx)
        
        # 3️⃣ Accumulate Perplexity answer text (skip only placeholder text)
        if pplx_answer and "Skipped in Fast Preview" not in pplx_answer:
            for brand in pplx_mentions.keys():
                existing_answer = pplx_results[brand].get("answer", "")
                pplx_results[brand]["answer"] = (
                    existing_answer + " " + pplx_answer
                ).strip()
        # 🔥 Early exit if Lumio signal stabilizes
        if i >= 4:
            top_serp_brands = sorted(
                serp_results.items(),
                key=lambda x: x[1]["mentions"],
                reverse=True
            )[:5]
        
            if "Lumio" in [b[0] for b in top_serp_brands]:
                break
                

        answers.append({
            "keyword": q["query"],
        
            # Explicit questions
            "chatgpt_question": primary_question,
            "perplexity_question": pplx_questions,
        
            # Answers
            "chatgpt_answer": chatgpt_answer_text,
            "perplexity_answer": pplx_answer,
        
            "chatgpt_sources": gsrc,
            "perplexity_sources": list(set(pplx_answer_sources + pplx_question_sources)),
        
            "top_brands_serp": top_serp,
            "top_brands_pplx": top_pplx
        })
        
        progress.progress(int((i + 1) / total * 100))
        
        # 🔥 Run Perplexity deep research ONCE per brand (hard capped)
        # 🔥 Optional Perplexity deep research (Deep mode only)
        if analysis_mode == "Deep Research" and i >= 3:
            for brand in list(all_pplx_brands - PERPLEXITY_DEEP_CACHE.keys())[:2]:
                if brand not in PERPLEXITY_DEEP_CACHE:
                    PERPLEXITY_DEEP_CACHE[brand] = cached_perplexity_research(
                        brand, segment
                    )[:4000]
        
                existing = pplx_results[brand].get("answer", "")
                pplx_results[brand]["answer"] = (
                    existing + " " + PERPLEXITY_DEEP_CACHE[brand]
                ).strip()


    progress.empty()   
    # CRITICAL: Return must be OUTSIDE the for loop
    return {
        "serp": dict(serp_results),
        "perplexity": dict(pplx_results),
        "answers": answers,
        "timestamp": datetime.now().strftime("%Y-%W")

    }


def persist_weekly_snapshot(res):
    rows = []

    for brand, data in res["serp"].items():
        rows.append({
            "Week": res["timestamp"],
            "Source": "SERP+ChatGPT",
            "Brand": brand,
            "Visibility": data.get("mentions", 0),
            "Sentiment": safe_avg_sentiment(data.get("sentences", []))
        })

    for brand, data in res["perplexity"].items():
        rows.append({
            "Week": res["timestamp"],
            "Source": "Perplexity",
            "Brand": brand,
            "Visibility": data.get("mentions", 0),
            "Sentiment": safe_avg_sentiment(data.get("sentences", []))
        })

    df = pd.DataFrame(rows)

    if os.path.exists(HISTORY_FILE):
        existing = load_history()
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(
            subset=["Week", "Source", "Brand"],
            keep="last",
            inplace=True
        )
        combined.to_csv(HISTORY_FILE, index=False)

    else:
        df.to_csv(HISTORY_FILE, index=False)
        
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    return pd.read_csv(HISTORY_FILE)     

def compute_weekly_delta(history_df, current_week):
    weeks = sorted(history_df["Week"].unique())
    prev_week = weeks[-2] if len(weeks) >= 2 else None

    if not prev_week:
        return pd.DataFrame()

    curr = history_df[history_df["Week"] == current_week]
    prev = history_df[history_df["Week"] == prev_week]

    merged = curr.merge(
        prev,
        on=["Brand", "Source"],
        suffixes=("_curr", "_prev")
    )

    merged["Visibility Δ"] = merged["Visibility_curr"] - merged["Visibility_prev"]
    merged["Sentiment Δ"] = merged["Sentiment_curr"] - merged["Sentiment_prev"]

    return merged    

        
# ================== DATAFRAME ==================
def to_df(data, source_count=1):
    df = pd.DataFrame(columns=[
        "Brand",
        "Visibility",
        "Sentiment Index",
        "Risk Index",
        "Confidence"
    ])

    if not data:
        return df

    rows = []
    for b, d in data.items():
        visibility = d.get("mentions", 0)
        sentences = d.get("sentences", [])

        complaints = d.get("complaints", 0)
        praise = d.get("praise", 0)

        rows.append({
            "Brand": b,
            "Visibility": visibility,
            "Sentiment Index": compute_sentiment_index(sentences),
            "Risk Index": compute_risk_index(complaints, praise),
            "Confidence": compute_confidence(
                visibility,
                len(sentences),
                source_count
            )
        })

    return pd.DataFrame(rows)

def compute_confidence(visibility, sentence_count, source_count):
    """
    Confidence heuristic:
    - Visibility weight
    - Sample size weight
    - Multi-source bonus
    """
    score = 0

    score += min(visibility * 5, 40)
    score += min(sentence_count * 2, 40)
    score += 20 if source_count > 1 else 0

    return min(score, 100)

def confidence_label(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


def build_lumio_diagnostics(res):
    serp_df = to_df(res["serp"])
    pplx_df = to_df(res["perplexity"])

    lumio_serp = serp_df[serp_df["Brand"] == "Lumio"]
    lumio_pplx = pplx_df[pplx_df["Brand"] == "Lumio"]

    competitors_serp = serp_df[serp_df["Brand"] != "Lumio"]
    competitors_pplx = pplx_df[pplx_df["Brand"] != "Lumio"]
    

    diagnostics = {}
    
    diagnostics["lumio_complaints"] = res["perplexity"].get("Lumio", {}).get("complaints", 0)
    diagnostics["lumio_praise"] = res["perplexity"].get("Lumio", {}).get("praise", 0)


    # --- SERP VISIBILITY ---
    if lumio_serp.empty and not competitors_serp.empty:
        diagnostics["serp_visibility_gap"] = None
        diagnostics["serp_visibility_status"] = "not_visible"
    else:
        diagnostics["serp_visibility_gap"] = (
            competitors_serp["Visibility"].mean() - lumio_serp["Visibility"].values[0]
            if not lumio_serp.empty and not competitors_serp.empty
            else 0
        )
        diagnostics["serp_visibility_status"] = "visible"
    
    # --- PERPLEXITY VISIBILITY ---
    if lumio_pplx.empty and not competitors_pplx.empty:
        diagnostics["pplx_visibility_gap"] = None
        diagnostics["pplx_visibility_status"] = "not_visible"
    else:
        diagnostics["pplx_visibility_gap"] = (
            competitors_pplx["Visibility"].mean() - lumio_pplx["Visibility"].values[0]
            if not lumio_pplx.empty and not competitors_pplx.empty
            else 0
        )
        diagnostics["pplx_visibility_status"] = "visible"

    # --- SERP SENTIMENT ---
    if lumio_serp.empty or not res["serp"].get("Lumio", {}).get("sentences"):
        diagnostics["serp_sentiment_gap"] = None
        diagnostics["serp_sentiment_status"] = "not_available"
    else:
        diagnostics["serp_sentiment_gap"] = (
            competitors_serp["Sentiment"].mean() - lumio_serp["Sentiment"].values[0]
        )
        diagnostics["serp_sentiment_status"] = "available"
    
    
    # --- PERPLEXITY SENTIMENT ---
    if lumio_pplx.empty or not res["perplexity"].get("Lumio", {}).get("sentences"):
        diagnostics["pplx_sentiment_gap"] = None
        diagnostics["pplx_sentiment_status"] = "not_available"
    else:
        diagnostics["pplx_sentiment_gap"] = (
            competitors_pplx["Sentiment"].mean() - lumio_pplx["Sentiment"].values[0]
        )
        diagnostics["pplx_sentiment_status"] = "available"

    # Weak sentiment words (SERP + Perplexity)
    lumio_sentences = (
        res["serp"].get("Lumio", {}).get("sentences", []) +
        res["perplexity"].get("Lumio", {}).get("sentences", [])
    )

    diagnostics["lumio_sentences"] = [
        s if isinstance(s, str) else s[0]
        for s in lumio_sentences[:50]
    ]

    # Top competing brands
    diagnostics["top_competitors"] = (
        competitors_serp
        .sort_values("Visibility", ascending=False)
        .head(3)["Brand"]
        .tolist()
        if not competitors_serp.empty
        else []
    )
    
    if not lumio_serp.empty and not lumio_pplx.empty:
        diagnostics["confidence_note"] = "High confidence (multi-source validation)"
    else:
        diagnostics["confidence_note"] = "Low confidence (limited visibility)"
    # -------------------------------
    # CONFIDENCE ALERT LOGIC
    # -------------------------------
    lumio_visibility = (
        serp_df[serp_df["Brand"] == "Lumio"]["Visibility"].sum()
        if not serp_df.empty else 0
    )
    
    lumio_sentence_count = (
        len(res["serp"].get("Lumio", {}).get("sentences", [])) +
        len(res["perplexity"].get("Lumio", {}).get("sentences", []))
    )

    
    diagnostics["lumio_confidence_score"] = compute_confidence(
        lumio_visibility,
        lumio_sentence_count,
        2 if diagnostics["pplx_visibility_status"] == "visible" else 1
    )
    
    diagnostics["low_confidence"] = (
        diagnostics["lumio_confidence_score"] < CONFIDENCE_ALERT_THRESHOLD
    )    
    
    return diagnostics


# ================== WORD CLOUD ==================
def render_wordcloud(sentences):
    text = " ".join([
        s if isinstance(s, str) else s[0]
        for s in sentences
    ])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
    
def render_brand_wordcloud(res, brand, source="Both"):
    sentences = []

    if source in ("SERP", "Both"):
        sentences.extend(
            res["serp"].get(brand, {}).get("sentences", [])
        )

    if source in ("Perplexity", "Both"):
        sentences.extend(
            res["perplexity"].get(brand, {}).get("sentences", [])
        )

    if not sentences:
        st.info(f"No sentences available for {brand} ({source}).")
        return

    render_wordcloud(sentences)    


# ================== DATA PROCESSING FOR MATRIX ==================
def prepare_matrix_dfs(target_brands, res):
    google_rows = []
    perplexity_rows = []
    
    for b in target_brands:
        # 1. Process Google Data
        g_sentences = res["serp"].get(b, {}).get("sentences", [])
        g_scores = extract_competitive_insights(b, g_sentences)
        google_rows.append({"Brand": b, **g_scores})

        
        # 2. Process Perplexity Data
        p_text = res["perplexity"].get(b, {}).get("answer", "")

        # Only extract if deep Perplexity research exists
        if p_text:
            p_scores = extract_perplex_competitive_insights_from_text(b, p_text)
        else:
            p_scores = {f: "N/A" for f in EXPANDED_FEATURES}
        
        perplexity_rows.append({"Brand": b, **p_scores})



    # Define the dataframes that were previously missing
    df_google = pd.DataFrame(google_rows).set_index("Brand")
    df_perplexity = pd.DataFrame(perplexity_rows).set_index("Brand")
    
    return df_google, df_perplexity

def export_dataframe(df, filename_prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {filename_prefix} (CSV)",
        data=csv,
        file_name=f"{filename_prefix}_{timestamp}.csv",
        mime="text/csv"
        
    )

def color_status(val):
    if val == "Excellent":
        return "background-color: #d4edda"   # light green
    elif val == "Good":
        return "background-color: #e2f0d9"
    elif val == "Average":
        return "background-color: #fff3cd"   # light yellow
    elif val == "Poor":
        return "background-color: #f8d7da"   # light red
    elif val == "Unknown" or val == "N/A":
        return "background-color: #f0f0f0"   # grey
    return ""

def render_tab6_lumio_recommendations(res):
    diagnostics = build_lumio_diagnostics(res)
    st.header("🎯 Strategic Recommendations for Lumio")
    st.metric("Lumio Complaints", diagnostics["lumio_complaints"])
    st.metric("Lumio Praise", diagnostics["lumio_praise"])

    st.caption(
        "Derived from market visibility (SERP + ChatGPT), expert consensus (Perplexity), "
        "and comparative sentiment signals."
    )
    st.caption(f"📊 Confidence Assessment: **{diagnostics.get('confidence_note','N/A')}**")
    if diagnostics.get("low_confidence"):
        st.warning(
            "⚠️ **Low Confidence Alert**: Lumio has insufficient visibility and/or sentiment data. "
            "Insights below may be directional only. Primary recommendation is to **increase visibility** "
            "across search and expert platforms."
        )

    

    # -------------------------------
    # VISIBILITY & SENTIMENT METRICS
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)

    # ---- SERP + ChatGPT VISIBILITY ----
    with col1:
        if diagnostics["serp_visibility_status"] == "not_visible":
            st.metric(
                "ChatGPT Visibility",
                "Not visible at all",
                help="Lumio does not appear in Google SERP or ChatGPT-generated answers"
            )
        else:
            st.metric(
                "ChatGPT Visibility Gap",
                f"{diagnostics['serp_visibility_gap']:.2f}",
                help="Difference in brand mentions vs market average"
            )

    # ---- SERP + ChatGPT SENTIMENT ----
    with col2:
        if diagnostics["serp_sentiment_status"] == "not_available":
            st.metric(
                "ChatGPT Sentiment",
                "Sentiment not found",
                help="Insufficient Lumio mentions to infer sentiment"
            )
        else:
            st.metric(
                "ChatGPT Sentiment Gap",
                f"{diagnostics['serp_sentiment_gap']:.2f}",
                help="Difference in average sentiment vs competitors"
            )

    # ---- PERPLEXITY VISIBILITY ----
    with col3:
        if diagnostics["pplx_visibility_status"] == "not_visible":
            st.metric(
                "Perplexity Visibility",
                "Not visible at all",
                help="Lumio does not appear in Perplexity expert responses"
            )
        else:
            st.metric(
                "Perplexity Visibility Gap",
                f"{diagnostics['pplx_visibility_gap']:.2f}",
                help="Difference in expert mentions vs competitors"
            )

    # ---- PERPLEXITY SENTIMENT ----
    with col4:
        if diagnostics["pplx_sentiment_status"] == "not_available":
            st.metric(
                "Perplexity Sentiment",
                "Sentiment not found",
                help="Insufficient expert commentary to infer sentiment"
            )
        else:
            st.metric(
                "Perplexity Sentiment Gap",
                f"{diagnostics['pplx_sentiment_gap']:.2f}",
                help="Difference in expert sentiment vs competitors"
            )

    # -------------------------------
    # INTERPRETATION CALLOUT
    # -------------------------------
    if (
        diagnostics["serp_visibility_status"] == "not_visible"
        and diagnostics["pplx_visibility_status"] == "not_visible"
    ):
        st.warning(
            "⚠️ Lumio is currently **not visible** across both consumer search (SERP/ChatGPT) "
            "and expert research (Perplexity). Recommendations will prioritize **visibility creation**."
        )

    elif diagnostics["serp_visibility_status"] == "visible" and diagnostics["pplx_visibility_status"] == "not_visible":
        st.info(
            "ℹ️ Lumio is visible in consumer-facing search but underrepresented in expert discourse. "
            "Recommendations will focus on **credibility, reviews, and expert validation**."
        )

    # -------------------------------
    # STRATEGIC RECOMMENDATIONS
    # -------------------------------
    with st.spinner("Generating Lumio-specific recommendations..."):
        if diagnostics.get("low_confidence"):
            st.info(
                "📌 Recommendations are **visibility-first** due to limited signal strength."
            )
        recommendations = generate_lumio_recommendations(diagnostics)


    st.markdown(recommendations)


# ================== UI ==================
st.title("📺 TV Brand Visibility Analyzer – India")

st.sidebar.header("🎛 Research Controls")
segment = st.sidebar.selectbox(
    "Price Segment",
    ["All", "Budget", "Mid", "Premium"]
)

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Fast Preview", "Deep Research"],
    help="Fast Preview skips Perplexity deep research for speed"
)



# =============================================================================
# st.sidebar.subheader("📡 API Usage (This Run)")
# st.sidebar.metric("SerpAPI Calls", API_COUNTERS["serpapi"])
# st.sidebar.metric("ChatGPT Calls", API_COUNTERS["chatgpt"])
# st.sidebar.metric("Perplexity Calls", API_COUNTERS["perplexity"])
# =============================================================================


SEGMENT_QUERIES = {
    "All": [
        "best smart tv in india 2026",
        "top tv brands in india",
        "best tv brand for indian households",
        "most reliable tv brand in india"
    ],

    "Budget": [
        "best tv under 30000 india",
        "best budget smart tv india",
        "best 43 inch tv under 30000",
        "best tv under 25000 india",
        "budget tv with good picture quality",
        "which budget tv has less lag"
    ],

    "Mid": [
        "best tv under 60000 india",
        "best 55 inch 4k tv india",
        "best mid range smart tv india",
        "best tv under 70000 india",
        "mid range tv with smooth ui",
        "best tv for movies under 70000"
    ],

    "Premium": [
        "best oled tv in india",
        "best premium tv brand india",
        "sony vs samsung premium tv",
        "best tv for ps5 india",
        "best tv with 144hz gaming india",
        "best tv for home theatre india"
    ]
}

PERCEPTION_QUERIES = [
    "which tv brand has least complaints in india",
    "which tv brand has least issues",
    "which tv brand is most value for money in india",
    "which tv brand has best after sales service in india",
    "which tv brand is most reliable long term",
    "which tv brand has fastest software"
]

segment_queries = SEGMENT_QUERIES.get(segment, [])
BASE_QUERIES = segment_queries + PERCEPTION_QUERIES

if st.button("🔄 Run Live Analysis"):
    # 1️⃣ Expand base queries using AI
    expanded_queries = generate_expanded_queries(segment, BASE_QUERIES)[:20]
    
    st.caption(f"🔍 Analyzing {len(expanded_queries)} queries")

    # 2️⃣ Normalize for analysis loop
    queries = [{"query": q.lower(), "volume": 100} for q in expanded_queries]

    st.session_state.results = run_analysis(queries)
    persist_weekly_snapshot(st.session_state.results)

if "results" in st.session_state:
    res = st.session_state.results

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visibility & Sentiment",
        "🧠 Voice of Customer",
        "🥇 Top Brands per Question",
        "📋 Competitive Matrix",
        "📆 Week-over-Week Trends",
        "🎯 Lumio Recommender"
    ])

    with tab1:
        st.header("📊 Visibility & Sentiment by Source")

        col1, col2 = st.columns(2)

    # ---- ChatGPT + SERP ----
    with col1:
        st.subheader("Google SERP + ChatGPT")

        df_serp = to_df(res["serp"], source_count = 1)
        if df_serp.empty:
            st.warning("No SERP/ChatGPT brand mentions found.")
        else:
            fig_serp = make_subplots(specs=[[{"secondary_y": True}]])
            fig_serp.add_bar(
                x=df_serp["Brand"],
                y=df_serp["Visibility"],
                name="Visibility"
            )
            fig_serp.add_scatter(
                x=df_serp["Brand"],
                y=df_serp["Sentiment Index"],
                secondary_y=True,
                mode="lines+markers",
                name="Sentiment Index"
            )
            fig_serp.update_layout(title="SERP + ChatGPT")
            st.plotly_chart(fig_serp, use_container_width=True)
            
    
    # ---- Perplexity ----
    with col2:
        st.subheader("Perplexity AI")

        df_pplx = to_df(res["perplexity"], source_count = 2)
        if df_pplx.empty:
            st.warning("No Perplexity brand mentions found.")
        else:
            fig_pplx = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pplx.add_bar(
                x=df_pplx["Brand"],
                y=df_pplx["Visibility"],
                name="Visibility"
            )
            fig_pplx.add_scatter(
                x=df_pplx["Brand"],
                y=df_pplx["Sentiment Index"],
                secondary_y=True,
                mode="lines+markers",
                name="Sentiment Index"
            )
            fig_pplx.update_layout(title="Perplexity")
            st.plotly_chart(fig_pplx, use_container_width=True)
            
            st.subheader("📤 Export Visibility Data")

            export_dataframe(df_serp, "serp_visibility")
            export_dataframe(df_pplx, "perplexity_visibility")
    st.subheader("⚠️ Brand Risk Index")

    risk_df = df_pplx[["Brand", "Risk Index"]].sort_values(
        "Risk Index", ascending=False
    )
    
    st.dataframe(
        risk_df.style.format({"Risk Index": "{:.2f}"})
                 .background_gradient(cmap="Reds"),
        use_container_width=True
    )
   
    if "results" in st.session_state:
        res = st.session_state.results
        
    with tab2:
        st.header("❓ Actual Questions Answered by Each Model")

        q_df = pd.DataFrame([
            {
                "Keyword": a["keyword"],
                "ChatGPT – Question Interpreted": a["chatgpt_question"],
                "Perplexity – Question Interpreted": a["perplexity_question"]
            }
            for a in res["answers"]
        ])
        
        st.dataframe(q_df, use_container_width=True)
        
        
        st.header("🧠 Model Answers & Evidence")
        # This is where your 'tab_questions' logic goes!
        for a in res["answers"]:
            with st.expander(f"Intent for: {a['keyword']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("ChatGPT (Inferred)")
                    st.info(a.get("chatgpt_answer", "N/A"))
                    st.caption("**Sources:**")
                    for link in a.get("chatgpt_sources", [])[:3]:
                        st.markdown(f"🔗 [{link[:30]}...]({link})")
                with col2:
                    st.subheader("Perplexity (Live)")
                    st.write(a["perplexity_question"])
                    st.success(a["perplexity_answer"])
                
                    if a["perplexity_sources"]:
                        st.caption("**Sources (Perplexity):**")
                        for i, link in enumerate(a["perplexity_sources"], 1):
                            st.markdown(f"{i}. 🔗 [{link}]({link})")
        
        st.subheader("☁️ Voice of Customer – Word Cloud")

        col1, col2 = st.columns(2)
        
        # --- SERP / ChatGPT Word Cloud ---
        with col1:
            st.markdown("**Google SERP + ChatGPT**")
            serp_sentences_all = []
            for v in res["serp"].values():
                serp_sentences_all.extend(v.get("sentences", []))
        
            if serp_sentences_all:
                render_wordcloud(serp_sentences_all)
            else:
                st.info("No SERP sentences available for word cloud.")
        
        # --- Perplexity Word Cloud ---
        with col2:
            st.markdown("**Perplexity (Expert Voice)**")
            pplx_sentences_all = []
            for v in res["perplexity"].values():
                pplx_sentences_all.extend(v.get("sentences", []))
        
            if pplx_sentences_all:
                render_wordcloud(pplx_sentences_all)
            else:
                st.info("No Perplexity sentences available for word cloud.")
                        
        st.header("☁️ Brand-specific Voice of Customer")

        # Collect all brands dynamically
        available_brands = sorted(
            set(list(res["serp"].keys()) + list(res["perplexity"].keys()))
        )
        
        if not available_brands:
            st.info("No brands available for word cloud.")
        else:
            col1, col2, col3 = st.columns([2, 2, 4])
        
            with col1:
                selected_brand = st.selectbox(
                    "Select Brand",
                    available_brands,
                    index=available_brands.index("Lumio") if "Lumio" in available_brands else 0
                )
        
            with col2:
                selected_source = st.selectbox(
                    "Source",
                    ["Both", "SERP", "Perplexity"]
                )
        
            with col3:
                st.caption(
                    "Word cloud generated from actual sentences where the selected brand "
                    "is mentioned. Reflects consumer (SERP/ChatGPT) and/or expert (Perplexity) language."
                )
        
            render_brand_wordcloud(
                res,
                brand=selected_brand,
                source=selected_source
            )                
        qa_export_df = pd.DataFrame([
        {
            "Keyword": a["keyword"],
            "ChatGPT Question": a["chatgpt_question"],
            "ChatGPT Answer": a["chatgpt_answer"],
            "ChatGPT Sources": ", ".join(a["chatgpt_sources"]),
            "Perplexity Question": a["perplexity_question"],
            "Perplexity Answer": a["perplexity_answer"],
            "Perplexity Sources": ", ".join(a["perplexity_sources"])
        }
        for a in res["answers"]
        ])

        st.subheader("📤 Export Q&A Dataset")
        export_dataframe(qa_export_df, "questions_answers")
              
    with tab3:
        st.header("🥇 Top Brands by Question & Source")

        for a in res["answers"]:
            st.subheader(a["keyword"].title())
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.markdown("**ChatGPT + Google SERP**")
                for b in a["top_brands_serp"]:
                    st.markdown(f"🏆 {b}")
    
            with col2:
                st.markdown("**Perplexity AI**")
                for b in a["top_brands_pplx"]:
                    st.markdown(f"🏆 {b}")

    with tab4:
        st.header("📋 2026 Competitive Matrix (Separate Intelligence Views)")
        st.caption(
            "Two independent perspectives: "
            "Market perception (SERP + ChatGPT) vs Expert consensus (Perplexity)."
        )
        st.caption(
            "⚠ Brands with **Low Confidence** have insufficient visibility or evidence. "
            "Comparisons should be treated as directional, not definitive."
        )

    
        # -------------------------------------------------
        # Identify brands to compare
        # -------------------------------------------------
        all_top_brands = set()
        for a in res["answers"]:
            all_top_brands.update(a["top_brands_serp"])
    
        target_brands = ["Lumio"] + [b for b in all_top_brands if b != "Lumio"]
    
        # -------------------------------------------------
        # 🟦 SERP + ChatGPT MATRIX (Market Perception)
        # -------------------------------------------------
        st.subheader("🟦 SERP + ChatGPT Competitive Matrix (Market Perception)")
    
        serp_matrix_rows = []
    
        with st.spinner("Analyzing SERP-based brand features..."):
            for b in target_brands:
                serp_sentences = res["serp"].get(b, {}).get("sentences", [])
                serp_scores = extract_competitive_insights(b, serp_sentences)
                
                confidence_score = compute_confidence(
                    visibility=res["serp"].get(b, {}).get("mentions", 0),
                    sentence_count=len(serp_sentences),
                    source_count=1
                )

                serp_matrix_rows.append({
                    "Brand": b,
                    **serp_scores,
                    "Confidence Score": confidence_score,
                    "Confidence Level": confidence_label(confidence_score)
                })
    
        df_serp_matrix = pd.DataFrame(serp_matrix_rows)
        
        df_serp_matrix = df_serp_matrix.sort_values(
            by="Confidence Score", ascending=False
        )

    
        st.dataframe(df_serp_matrix, use_container_width=True)
    
        export_dataframe(df_serp_matrix, "competitive_matrix_serp")
    
        # -------------------------------------------------
        # 🟪 PERPLEXITY MATRIX (Expert Consensus)
        # -------------------------------------------------
        st.subheader("🟪 Perplexity Competitive Matrix (Expert Consensus)")

        pplx_matrix_rows = []
        
        with st.spinner("Analyzing Perplexity expert research..."):
            for b in target_brands:
                p_text = (
                    res["perplexity"].get(b, {}).get("answer", "")
                    or " ".join(res["perplexity"].get(b, {}).get("sentences", []))
                )

        
                if p_text:
                    if p_text and len(p_text.split()) > 40:
                        pplx_scores = extract_perplex_competitive_insights_from_text(b, p_text)
                    else:
                        pplx_scores = {f: "Unknown" for f in EXPANDED_FEATURES}
                    sentence_count = len(p_text.split("."))
                else:
                    pplx_scores = {f: "N/A" for f in EXPANDED_FEATURES}
                    sentence_count = 0
                
                confidence_score = compute_confidence(
                    visibility=res["perplexity"].get(b, {}).get("mentions", 0),
                    sentence_count=sentence_count,
                    source_count=2
                )

                pplx_matrix_rows.append({
                    "Brand": b,
                    **pplx_scores,
                    "Confidence Score": confidence_score,
                    "Confidence Level": confidence_label(confidence_score)
                })

        
        df_pplx_matrix = pd.DataFrame(pplx_matrix_rows)
        
        df_pplx_matrix = df_pplx_matrix.sort_values(
            by="Confidence Score", ascending=False
        )

        
        st.dataframe(df_pplx_matrix, use_container_width=True)
        
        export_dataframe(df_pplx_matrix, "competitive_matrix_perplexity")
        
    with tab5:
        st.header("📆 Week-over-Week Brand Movement")
    
        history_df = load_history()

        if history_df.empty or history_df["Week"].nunique() < 2:
            st.info("Week-over-week comparison will appear once at least two weekly runs are available.")
        else:
            delta_df = compute_weekly_delta(history_df, res["timestamp"])
    
            st.subheader("📈 Visibility Change (WoW)")
            st.dataframe(
                delta_df[
                    ["Brand", "Source", "Visibility_curr", "Visibility_prev", "Visibility Δ"]
                ].sort_values("Visibility Δ", ascending=False),
                use_container_width=True
            )
    
            st.subheader("💬 Sentiment Change (WoW)")
            st.dataframe(
                delta_df[
                    ["Brand", "Source", "Sentiment_curr", "Sentiment_prev", "Sentiment Δ"]
                ].sort_values("Sentiment Δ", ascending=False),
                use_container_width=True
            )
    
    with tab6:
        render_tab6_lumio_recommendations(res)
