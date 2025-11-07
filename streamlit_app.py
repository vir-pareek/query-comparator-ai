import streamlit as st
import sqlite3
import joblib
import os

# Local imports
from src.extract_features import count_kw, plan_feats


# ----------------------------
# Paths
# ----------------------------

DB_PATH = os.path.join("data", "synth.db")
REG_PATH = "models/regressor.joblib"
CLS_PATH = "models/pairwise_clf.joblib"


# ----------------------------
# Load ML models
# ----------------------------
@st.cache_resource
def load_models():
    reg_model, reg_cols = joblib.load(REG_PATH)
    cls_model, cls_cols = joblib.load(CLS_PATH)
    return reg_model, reg_cols, cls_model, cls_cols


reg_model, reg_cols, cls_model, cls_cols = load_models()


# ----------------------------
# Helper: Run EXPLAIN QUERY PLAN
# ----------------------------

def explain(db, sql):
    cur = db.cursor()
    cur.execute("EXPLAIN QUERY PLAN " + sql)
    rows = cur.fetchall()
    # Combine all plan steps
    return " | ".join([r[3] for r in rows if len(r) >= 4])


# ----------------------------
# Helper: Predict latency + winner
# ----------------------------

def predict_pair(a_sql, b_sql):

    db = sqlite3.connect(DB_PATH)

    # Query plans
    a_plan = explain(db, a_sql)
    b_plan = explain(db, b_sql)

    # Feature extraction (dict)
    fa = count_kw(a_sql)
    fa.update(plan_feats(a_plan))

    fb = count_kw(b_sql)
    fb.update(plan_feats(b_plan))

    # Convert feature dicts to ordered lists
    def to_vector(feat_dict, order):
        return [feat_dict.get(col, 0) for col in order]

    a_vec = [to_vector(fa, reg_cols)]
    b_vec = [to_vector(fb, reg_cols)]

    # Regression prediction
    a_lat = float(reg_model.predict(a_vec)[0])
    b_lat = float(reg_model.predict(b_vec)[0])

    # Classification: compute diff vector
    diff = {}
    for k in fa:
        diff["diff_" + k] = fa.get(k, 0) - fb.get(k, 0)

    diff["diff_uses_index"] = (
        (1 if "using index" in a_plan.lower() else 0) -
        (1 if "using index" in b_plan.lower() else 0)
    )

    diff_vec = [diff.get(col, 0) for col in cls_cols]
    winner = cls_model.predict([diff_vec])[0]

    return a_plan, b_plan, a_lat, b_lat, winner


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("🔍 QueryOptimizerAI")
st.write("Compare two SQL queries and predict which one is faster using ML.")

sql_a = st.text_area("SQL Query A", height=160)
sql_b = st.text_area("SQL Query B", height=160)

if st.button("Compare Queries"):

    if not sql_a.strip() or not sql_b.strip():
        st.error("Please enter both SQL queries.")
    else:
        with st.spinner("Analyzing..."):
            a_plan, b_plan, a_lat, b_lat, winner = predict_pair(sql_a, sql_b)

        st.subheader("📌 Query Plans")
        st.write("**Plan A:**", a_plan)
        st.write("**Plan B:**", b_plan)

        st.subheader("⏱ Predicted Latency (log-scale)")
        st.write(f"✅ Query A: `{a_lat:.4f}`")
        st.write(f"✅ Query B: `{b_lat:.4f}`")

        st.subheader("🏆 Verdict")
        if winner == 1:
            st.success("✅ Query **A** is predicted to be faster")
        else:
            st.success("✅ Query **B** is predicted to be faster")