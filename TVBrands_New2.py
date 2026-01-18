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

def perplexity_question_miner(keyword):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "user",
                "content": f"List the most common real-world questions Indian consumers ask online when researching about {keyword}?"
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
            return "Perplexity questions unavailable.", []

        data = r.json()

        questions = data["choices"][0]["message"]["content"]

        sources = [
            c["url"] for c in data.get("citations", [])
            if isinstance(c, dict) and "url" in c
        ]

        return questions, sources

    except Exception as e:
        # ❗ HARD GUARD 2 — never crash Streamlit
        return f"Perplexity error: {str(e)}", []


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
                    mentions[brand]["sentences"].append(t)
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



# ================== LLM-POWERED EXTRACTION OF FEATURES ==================
def extract_competitive_insights(brand, sentences):
    """Refined to specifically target Boot-up Speed and UI performance."""
    if not sentences:
        return {f: "N/A" for f in EXPANDED_FEATURES}
    
    context = " ".join(sentences)[:4000]
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
- Chat GPT Visibility gap vs competitors: {diagnostics["serp_visibility_gap"]:.2f}
- Chat GPT Sentiment gap vs competitors: {diagnostics["serp_sentiment_gap"]:.2f}
- Perplexity Visibility gap vs competitors: {diagnostics["pplx_visibility_gap"]:.2f}
- Perplexity Sentiment gap vs competitors: {diagnostics["pplx_sentiment_gap"]:.2f}
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

# ================== UPDATED ANALYSIS LOOP ==================
def run_analysis(queries):
    # Initialize the storage dictionaries that were missing
    serp_results = defaultdict(lambda: {"mentions": 0, "sentences": []})
    pplx_results = defaultdict(lambda: {"mentions": 0, "sentences": []})
    answers = []

    for q in queries[:MAX_SERP_QUERIES]:
        # 1. Get SERP Data
        serp_data = serp_search(q["query"])
        
        # 2. Extract Brand Mentions from SERP
        serp_mentions = extract_mentions([d["text"] for d in serp_data])
        for brand, data in serp_mentions.items():
            serp_results[brand]["mentions"] += data["count"]
            serp_results[brand]["sentences"].extend(data["sentences"])

        # 3. Get Questions & Sources
        chatgpt_q, gsrc = chatgpt_question_generator(q["query"], serp_data)
        chatgpt_mentions = extract_mentions([chatgpt_q])
        primary_question = chatgpt_q.strip().split("\n")[0]
        chatgpt_answer_text = chatgpt_answer_question(
            primary_question,
            context=" ".join([d["text"] for d in serp_data[:3]])
        )
        for brand, data in chatgpt_mentions.items():
            serp_results[brand]["mentions"] += data["count"]
            serp_results[brand]["sentences"].extend(data["sentences"])       
    

        # 4. Simple logic to find "Top Brands" for this specific query
        # We sort brands found in this specific search by mention count
        sorted_serp = sorted(serp_mentions.items(), key=lambda x: x[1]["count"], reverse=True)
        top_serp = [b[0] for b in sorted_serp[:3]]
        
        if not PERPLEXITY_API_KEY:
            pplx_answer = "Perplexity disabled."
            pplx_questions = "Perplexity disabled."
            pplx_sources = []
            pplx_mentions = {}
        else:
            pplx_answer, pplx_answer_sources = perplexity_answer(q["query"])
            pplx_questions, pplx_question_sources = perplexity_question_miner(q["query"])
            pplx_sources = pplx_answer_sources
            pplx_mentions = extract_mentions([pplx_answer])
            
            
        # 1️⃣ Aggregate basic Perplexity signals
        for brand, data in pplx_mentions.items():
            pplx_results[brand]["mentions"] += data["count"]
            pplx_results[brand]["sentences"].extend(data["sentences"])
        
        # 2️⃣ Decide Top-3 Perplexity brands for THIS query
        sorted_pplx = sorted(
            pplx_mentions.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        top_pplx = [b[0] for b in sorted_pplx[:3]]
        
        # 3️⃣ Run deep research ONLY for Top-3 brands
        for brand in top_pplx:
            if brand not in PERPLEXITY_DEEP_CACHE:
                PERPLEXITY_DEEP_CACHE[brand] = run_perplexity_deep_research(
                    brand, segment
                )[:4000]
        
            pplx_results[brand]["answer"] = PERPLEXITY_DEEP_CACHE[brand]

        
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
            "Sentiment": round(
                sum(sentiment_score(s) for s in data.get("sentences", [])) /
                len(data.get("sentences", [])),
                3
            ) if data.get("sentences") else 0
        })

    for brand, data in res["perplexity"].items():
        rows.append({
            "Week": res["timestamp"],
            "Source": "Perplexity",
            "Brand": brand,
            "Visibility": data.get("mentions", 0),
            "Sentiment": round(
                sum(sentiment_score(s) for s in data.get("sentences", [])) /
                len(data.get("sentences", [])),
                3
            ) if data.get("sentences") else 0
        })

    df = pd.DataFrame(rows)

    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)
        
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    return pd.read_csv(HISTORY_FILE)     

def compute_weekly_delta(history_df, current_week):
    prev_week = (
        history_df["Week"]
        .drop_duplicates()
        .sort_values()
        .iloc[-2]
        if history_df["Week"].nunique() > 1
        else None
    )

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
def to_df(data):
    df = pd.DataFrame(columns=["Brand", "Visibility", "Sentiment"])
    if not data:
        return df

    rows = []
    for b, d in data.items():
        rows.append({
            "Brand": b,
            "Visibility": d.get("mentions", 0),
            "Sentiment": round(
                sum(sentiment_score(s) for s in d.get("sentences", [])) / len(d.get("sentences", [])),
                3
            ) if d.get("sentences") else 0
        })
    return pd.DataFrame(rows, columns=["Brand", "Visibility", "Sentiment"])

def build_lumio_diagnostics(res):
    serp_df = to_df(res["serp"])
    pplx_df = to_df(res["perplexity"])

    lumio_serp = serp_df[serp_df["Brand"] == "Lumio"]
    lumio_pplx = pplx_df[pplx_df["Brand"] == "Lumio"]

    competitors_serp = serp_df[serp_df["Brand"] != "Lumio"]
    competitors_pplx = pplx_df[pplx_df["Brand"] != "Lumio"]

    diagnostics = {}

    # Visibility gap
    diagnostics["serp_visibility_gap"] = (
        competitors_serp["Visibility"].mean() - lumio_serp["Visibility"].values[0]
        if not lumio_serp.empty and not competitors_serp.empty
        else 0
    )
    
    diagnostics["pplx_visibility_gap"] = (
        competitors_pplx["Visibility"].mean() - lumio_pplx["Visibility"].values[0]
        if not lumio_pplx.empty and not competitors_pplx.empty
        else 0
    )

    # Sentiment gap
    diagnostics["serp_sentiment_gap"] = (
        competitors_serp["Sentiment"].mean() - lumio_serp["Sentiment"].values[0]
        if not lumio_serp.empty and not competitors_serp.empty
        else 0
    )
    
    diagnostics["pplx_sentiment_gap"] = (
        competitors_pplx["Sentiment"].mean() - lumio_pplx["Sentiment"].values[0]
        if not lumio_pplx.empty and not competitors_pplx.empty
        else 0
    )

    # Weak sentiment words (SERP + Perplexity)
    lumio_sentences = (
        res["serp"].get("Lumio", {}).get("sentences", []) +
        res["perplexity"].get("Lumio", {}).get("sentences", [])
    )

    diagnostics["lumio_sentences"] = lumio_sentences[:50]

    # Top competing brands
    diagnostics["top_competitors"] = (
        competitors_serp
        .sort_values("Visibility", ascending=False)
        .head(3)["Brand"]
        .tolist()
        if not competitors_serp.empty
        else []
    )

    return diagnostics


# ================== WORD CLOUD ==================
def render_wordcloud(sentences):
    text = " ".join(sentences)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)


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
            p_scores = extract_perplex_competitive_insights(b, p_text)
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


# ================== UI ==================
st.title("📺 TV Brand Visibility Analyzer – India")

st.sidebar.header("🎛 Research Controls")
segment = st.sidebar.selectbox(
    "Price Segment",
    ["All", "Budget", "Mid", "Premium"]
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

        df_serp = to_df(res["serp"])
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
                y=df_serp["Sentiment"],
                secondary_y=True,
                mode="lines+markers",
                name="Sentiment"
            )
            fig_serp.update_layout(title="SERP + ChatGPT")
            st.plotly_chart(fig_serp, use_container_width=True)
            

    # ---- Perplexity ----
    with col2:
        st.subheader("Perplexity AI")

        df_pplx = to_df(res["perplexity"])
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
                y=df_pplx["Sentiment"],
                secondary_y=True,
                mode="lines+markers",
                name="Sentiment"
            )
            fig_pplx.update_layout(title="Perplexity")
            st.plotly_chart(fig_pplx, use_container_width=True)
            
            st.subheader("📤 Export Visibility Data")

            export_dataframe(df_serp, "serp_visibility")
            export_dataframe(df_pplx, "perplexity_visibility")

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
                serp_matrix_rows.append({"Brand": b, **serp_scores})
    
        df_serp_matrix = pd.DataFrame(serp_matrix_rows)
    
        st.dataframe(
            df_serp_matrix.style.applymap(
                color_status, subset=EXPANDED_FEATURES
            ),
            use_container_width=True
        )
    
        export_dataframe(df_serp_matrix, "competitive_matrix_serp")
    
        # -------------------------------------------------
        # 🟪 PERPLEXITY MATRIX (Expert Consensus)
        # -------------------------------------------------
        st.subheader("🟪 Perplexity Competitive Matrix (Expert Consensus)")
    
        pplx_matrix_rows = []
    
        with st.spinner("Analyzing Perplexity expert research..."):
            for b in target_brands:
                pplx_scores = extract_perplex_competitive_insights(b, segment)
                pplx_matrix_rows.append({"Brand": b, **pplx_scores})
    
        df_pplx_matrix = pd.DataFrame(pplx_matrix_rows)
    
        st.dataframe(
            df_pplx_matrix.style.applymap(
                color_status, subset=EXPANDED_FEATURES
            ),
            use_container_width=True
        )
    
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
        st.header("🎯 Strategic Recommendations for Lumio")
        st.caption("Derived from live market visibility, sentiment signals, and competitor benchmarks.")

        if "Lumio" not in res["serp"] and "Lumio" not in res["perplexity"]:
            st.warning("Not enough Lumio data available to generate recommendations.")
        else:
            diagnostics = build_lumio_diagnostics(res)
    
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                st.metric(
                    "ChatGPT Visibility Gap vs Market",
                    f"{diagnostics['serp_visibility_gap']:.2f}",
                    delta="Needs improvement" if diagnostics["serp_visibility_gap"] > 0 else "On par"
                )
    
            with col2:
                st.metric(
                    "ChatGPT Sentiment Gap vs Market",
                    f"{diagnostics['serp_sentiment_gap']:.2f}",
                    delta="Negative perception" if diagnostics["serp_sentiment_gap"] > 0.05 else "Healthy"
                )
                
            with col3:
                st.metric(
                    "Perplexity Visibility Gap vs Market",
                    f"{diagnostics['pplx_visibility_gap']:.2f}",
                    delta="Needs improvement" if diagnostics["pplx_visibility_gap"] > 0 else "On par"
                )
    
            with col2:
                st.metric(
                    "Perplexity Sentiment Gap vs Market",
                    f"{diagnostics['pplx_sentiment_gap']:.2f}",
                    delta="Negative perception" if diagnostics["pplx_sentiment_gap"] > 0.05 else "Healthy"
                )    
    
            with st.spinner("Generating Lumio-specific recommendations..."):
                recs = generate_lumio_recommendations(diagnostics)
    
            st.markdown(recs)

