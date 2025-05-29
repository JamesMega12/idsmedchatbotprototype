import streamlit as st
import pandas as pd
import requests, os, re, json, csv
import ollama
import operator
from langdetect import detect
from datetime import datetime
import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer

# ─── Page config & session defaults ───────────────────────────
st.set_page_config(layout="wide")
st.session_state.setdefault("history", [])
st.session_state.setdefault("api_calls", [])
st.session_state.setdefault("user_name", "")

# ─── Device & API settings ────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY           = os.getenv("DEEPSEEK_API_KEY", "sk-…")
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"
LOCAL_MODEL_NAME  = "deepseek-r1:1.5b"
LOG_CSV_PATH      = "api_activity.csv"

# ─── Helpers to find data files ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
def find_file_recursive(base_dir: str, exts: list[str]) -> str | None:
    for dp, _, fns in os.walk(base_dir):
        if any(skip in dp.lower() for skip in ("venv","site-packages",".chroma_db")):
            continue
        for fn in fns:
            if any(fn.lower().endswith(ext) for ext in exts):
                return os.path.join(dp, fn)
    return None

# ─── Load & clean stock data ─────────────────────────────────
@st.cache_data
def load_stock_data():
    for root in (DATA_DIR, BASE_DIR):
        p = find_file_recursive(root, [".xlsx", ".xls"])
        if p:
            df = pd.read_excel(p, header=None, engine="openpyxl")
            df = df.drop(0).reset_index(drop=True)
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df.drop(0).reset_index(drop=True)
            df = df.loc[:, ~df.columns.str.contains("Unnamed")]
            df = df.dropna(axis=1, how="all")
            return df, p
    return None, None

# ─── Load & normalize funnel data ─────────────────────────────
@st.cache_data
def load_funnel_data():
    for root in (DATA_DIR, BASE_DIR):
        p = find_file_recursive(root, [".parquet"])
        if p:
            df = pd.read_parquet(p)
            df["HREmployeeName_norm"] = df["HREmployeeName"].str.lower().str.strip()
            return df, p
    return None, None

# ─── Translation support ─────────────────────────────────────
try:
    @st.cache_data
    def load_translators():
        t1 = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
        m1 = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-id-en").to(DEVICE)
        t2 = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
        m2 = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-id").to(DEVICE)
        return t1, m1, t2, m2

    d_id2en_tok, d_id2en_mod, d_en2id_tok, d_en2id_mod = load_translators()
    translation_enabled = True
    def translate(txt, tok, mod):
        b = tok([txt], return_tensors="pt", padding=True).to(DEVICE)
        o = mod.generate(**b)
        return tok.batch_decode(o, skip_special_tokens=True)[0]
except ImportError:
    translation_enabled = False
    d_id2en_tok = d_id2en_mod = d_en2id_tok = d_en2id_mod = None
    def translate(txt, tok, mod): return txt

# ─── Simple math handler ──────────────────────────────────────
_math_ops = {"+":operator.add, "-":operator.sub, "*":operator.mul, "/":operator.truediv}
def math_handler(q:str) -> str|None:
    m = re.fullmatch(r"\s*([\d.]+)\s*([\+\-\*\/])\s*([\d.]+)\s*", q)
    if not m: return None
    a,op,b = float(m.group(1)), m.group(2), float(m.group(3))
    try: return str(_math_ops[op](a,b))
    except ZeroDivisionError: return "⚠️ Division by zero."

# ─── Logging & LLM wrappers ───────────────────────────────────
def log_remote_query(msgs, resp):
    rec = {
        "timestamp": datetime.now().isoformat(),
        "payload":   json.dumps(msgs, ensure_ascii=False),
        "response":  resp[:200],
        "stock_rows":  len(st.session_state.df_stock) if st.session_state.df_stock is not None else 0,
        "funnel_rows": len(st.session_state.df_funnel) if st.session_state.df_funnel is not None else 0
    }
    st.session_state.api_calls.append(rec)
    exists = os.path.isfile(LOG_CSV_PATH)
    with open(LOG_CSV_PATH,"a",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rec.keys())
        if not exists: w.writeheader()
        w.writerow(rec)

def deepseek_api_chat(msgs):
    r = requests.post(
        DEEPSEEK_CHAT_URL,
        headers={"Authorization":f"Bearer {API_KEY}", "Content-Type":"application/json"},
        json={"model":"deepseek-chat","messages":msgs,"stream":False}
    )
    return r.json().get("choices",[{}])[0]\
           .get("message",{}).get("content","⚠️ API error")

def chat_with_deepseek(msgs):
    if st.session_state.mode=="Local":
        try:
            return ollama.chat(model=LOCAL_MODEL_NAME, messages=msgs)["message"]["content"]
        except:
            st.sidebar.warning("⚠️ Ollama unreachable—switching to Remote.")
            st.session_state.mode="Remote"
    resp = deepseek_api_chat(msgs)
    log_remote_query(msgs, resp)
    return resp

# ─── Zero-shot intent setup ───────────────────────────────────
FUNNEL_INTENTS = {
  "count_funnels":        "Count how many funnels I have",
  "count_closed_funnels": "Count how many closed (Won) funnels I have",
  "count_closed_cases":   "Count how many closed (Won) cases I have",
  "top_n_funnels":        "Get the top N funnels by probability",
  "max_branch_closed":    "Which branch has the most closed (Won) cases",
  "max_prob_funnel":      "Which funnel has the highest probability",
  "recommend_my_funnel":  "Recommend a funnel for me"
}
@st.cache_resource
def load_intent_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
intent_clf = load_intent_classifier()
THRESHOLD = 0.6

# ─── Code-gen fallback ─────────────────────────────────────────
def answer_with_deepseek_code(q, df, history):
    cols   = df.columns.tolist()
    sample = df.head(10).to_dict("records")
    system = (
      "You are a pandas expert. `df` has columns:\n"
      f"{cols}\nFirst 10 rows:\n{sample}\n\n"
      "Write Python code to compute the answer into `result`."
    )
    msgs = [{"role":"system","content":system}] + history + [{"role":"user","content":q}]
    resp = chat_with_deepseek(msgs)
    m = re.search(r"```(?:python)?(.*?)```", resp, re.S)
    code = m.group(1) if m else resp
    local = {"df":df, "pd":pd}
    try:
        exec(code, {}, local)
        return str(local.get("result", resp))
    except:
        return general_llm_handler(q, history,
            df_funnel=(df if "FunnelID" in df.columns else None))

# ─── Funnel handlers ─────────────────────────────────────────
def handle_count_funnels(q, df, user_name, **_):
    cnt = (df["HREmployeeName_norm"] == user_name.lower()).sum()
    return f"You have {cnt} funnel(s)."

def handle_count_closed_funnels(q, df, user_name, **_):
    filt = df["FunnelStatus"].str.contains("Won", case=False)
    filt &= df["HREmployeeName_norm"].eq(user_name.lower())
    return f"You have {int(filt.sum())} Closed (Won) funnel(s)."

def handle_count_closed_cases(q, df, user_name, **_):
    filt = df["FunnelStatus"].str.contains("Won", case=False)
    return f"Total Closed (Won) cases: {int(filt.sum())}."

def handle_top_n_funnels(q, df, user_name, **_):
    m = re.search(r"top\s+(\d+)", q.lower())
    n = int(m.group(1)) if m else 3
    sel = df[df["HREmployeeName_norm"]==user_name.lower()]
    sel = sel[sel["FunnelStatus"].str.contains("Won", case=False)]
    if sel.empty: return "No closed funnels found for you."
    topn = sel.nlargest(n, "Probability Successful")
    return "\n".join(f"- {r['FunnelID']}: {r['Probability Successful']:.2f}"
                     for _,r in topn.iterrows())

def handle_max_branch_closed(q, df, user_name, **_):
    closed = df[df["FunnelStatus"].str.contains("Won", case=False)]
    vc = closed["SalesBranch"].value_counts()
    if vc.empty: return "No Closed (Won) cases."
    return f"{vc.idxmax()} has the most Closed (Won) cases ({vc.max()})."

def handle_max_prob_funnel(q, df, user_name, **_):
    idx = df["Probability Successful"].astype(float).idxmax()
    return f"FunnelID {df.at[idx,'FunnelID']} p={df.at[idx,'Probability Successful']:.2f}"

def handle_recommend_funnel(q, df, user_name, **_):
    sel = df[
        df["HREmployeeName_norm"].eq(user_name.lower()) &
        ~df["FunnelStatus"].str.contains("Closed", case=False)
    ]
    if sel.empty: return "No open funnels found for you."
    idx = sel["Probability Successful"].astype(float).idxmax()
    return f"You should go for {sel.at[idx,'FunnelID']} (p={sel.at[idx,'Probability Successful']:.2f})"

# ─── General LLM fallback ────────────────────────────────────
def general_llm_handler(q, history, df_funnel=None):
    sys = "You are a helpful assistant."
    if df_funnel is not None and "why" in q.lower():
        m = re.search(r"\b([A-Za-z0-9-]+)\b", q)
        if m and m.group(1) in df_funnel["FunnelID"].values:
            row = df_funnel[df_funnel["FunnelID"]==m.group(1)].iloc[0]
            attrs = {
              "Probability Successful": row["Probability Successful"],
              "FSSWinRate":             row["FSSWinRate"],
              "LoyaltyScore":           row["LoyaltyScore"],
              "SparkStageLocalDescription": row["SparkStageLocalDescription"]
            }
            sys += "\n\nAttributes:\n" + json.dumps(attrs, indent=2)
    msgs = [{"role":"system","content":sys}] + history + [{"role":"user","content":q}]
    return chat_with_deepseek(msgs)

# ─── Master router ────────────────────────────────────────────
def answer_question(q, df_stock, df_funnel, history, user_name):
    try: lang = detect(q)
    except: lang = "en"
    text = translate(q, d_id2en_tok, d_id2en_mod) \
           if translation_enabled and lang.startswith("id") else q

    # 1) math
    if (m := math_handler(text)) is not None:
        return m

    # 2) explicit top-N closed funnels
    if df_funnel is not None and re.search(r"top\s+\d+.*closed\s+funnels", text, re.I):
        return handle_top_n_funnels(text, df_funnel, user_name, history=history)

    # 3) stock Qs
    if df_stock is not None and re.search(r"\bstock\b|\bproduct\b|\bpo\b", text, re.I):
        return answer_with_deepseek_code(text, df_stock, history)

    # 4) funnel via zero-shot
    if df_funnel is not None:
        res   = intent_clf(text, list(FUNNEL_INTENTS.values()), multi_label=False)
        intent,score = res["labels"][0], res["scores"][0]
        if score > THRESHOLD:
            fn = {
              FUNNEL_INTENTS["count_funnels"]:        handle_count_funnels,
              FUNNEL_INTENTS["count_closed_funnels"]: handle_count_closed_funnels,
              FUNNEL_INTENTS["count_closed_cases"]:   handle_count_closed_cases,
              FUNNEL_INTENTS["top_n_funnels"]:        handle_top_n_funnels,
              FUNNEL_INTENTS["max_branch_closed"]:    handle_max_branch_closed,
              FUNNEL_INTENTS["max_prob_funnel"]:      handle_max_prob_funnel,
              FUNNEL_INTENTS["recommend_my_funnel"]:  handle_recommend_funnel
            }.get(intent)
            if fn:
                return fn(text, df_funnel, user_name, history=history)
        return answer_with_deepseek_code(text, df_funnel, history)

    # 5) final fallback
    return general_llm_handler(text, history, df_funnel)

# ─── Load data BEFORE UI ─────────────────────────────────────
df_stock,  stock_file  = load_stock_data()
df_funnel, funnel_file = load_funnel_data()
st.session_state.df_stock  = df_stock
st.session_state.df_funnel = df_funnel

# ─── Streamlit UI ────────────────────────────────────────────
st.title("📦 idsMED Stock & Funnel Chat")
st.markdown("👋 **Welcome to idsMED QA!**")

with st.sidebar:
    # editable name field
    name = st.text_input("Enter your HREmployeeName:", value=st.session_state.user_name)
    if name:
        st.session_state.user_name = name.strip()

    st.markdown(f"**User:** {st.session_state.user_name}")

    # validate against funnel data
    if df_funnel is not None:
        if st.session_state.user_name.lower() not in df_funnel["HREmployeeName_norm"].unique():
            st.warning("⚠️ Name not found in funnel data; chat will still proceed.")

    st.radio("Mode", ["Local","Remote"], key="mode")

    if df_stock is not None:
        st.success(f"Stock: `{os.path.relpath(stock_file,BASE_DIR)}` ({len(df_stock)} rows)")
    if df_funnel is not None:
        st.success(f"Funnel: `{os.path.relpath(funnel_file,BASE_DIR)}` ({len(df_funnel)} rows)")

    st.markdown("### API Activity (last 5)")
    for rec in st.session_state.api_calls[-5:]:
        st.text(f"{rec['timestamp']} → {rec['payload'][:50]}...")

    if os.path.exists(LOG_CSV_PATH):
        st.download_button("Download API log", open(LOG_CSV_PATH,"rb"),
                           file_name="api_activity.csv", mime="text/csv")

    # Refresh clears history & calls
    if st.button("🔄 Refresh Chat"):
        st.session_state.history   = []
        st.session_state.api_calls = []

# render chat history
for i,msg in enumerate(st.session_state.history):
    who = "You" if msg["role"]=="user" else "Bot"
    st.markdown(f"**{who}:** {msg['content']}")
    if who=="Bot" and st.button("Flag incorrect", key=f"flag_{i}"):
        prev = st.session_state.history[i-1]["content"] if i>0 else ""
        with open("flagged.csv","a",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            if f.tell()==0: w.writerow(["timestamp","question","answer"])
            w.writerow([datetime.now().isoformat(),prev,msg["content"]])
        st.info("Logged for review.")

# user input
with st.form("chat", clear_on_submit=True):
    user_input = st.text_input("Your message")
    send = st.form_submit_button("Send")

if send and user_input:
    st.session_state.history.append({"role":"user","content":user_input})
    reply = answer_question(
        user_input,
        df_stock,
        df_funnel,
        st.session_state.history,
        st.session_state.user_name
    )
    st.session_state.history.append({"role":"assistant","content":reply})
