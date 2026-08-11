import streamlit as st
import sqlite3
import joblib
import os
import re

# Local imports
from src.extract_features import count_kw, plan_feats


# ----------------------------
# Paths
# ----------------------------

DB_PATH = os.path.join("data", "synth.db")
REG_PATH = "models/regressor.joblib"
CLS_PATH = "models/pairwise_clf.joblib"


# ----------------------------
# Validate database exists and has tables
# ----------------------------

def validate_db():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at `{DB_PATH}`. Please run `python src/create_db.py` first.")
        st.stop()
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()
        required = {"users", "products", "orders", "reviews"}
        missing = required - set(tables)
        if missing:
            st.error(f"Database is missing tables: {missing}. Please run `python src/create_db.py` to regenerate.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

validate_db()


# ----------------------------
# Load ML models
# ----------------------------

@st.cache_resource
def load_models():
    if not os.path.exists(REG_PATH) or not os.path.exists(CLS_PATH):
        st.error("Model files not found. Please run `python src/train_model.py` first.")
        st.stop()
    reg_model, reg_cols = joblib.load(REG_PATH)
    cls_model, cls_cols = joblib.load(CLS_PATH)
    return reg_model, reg_cols, cls_model, cls_cols


reg_model, reg_cols, cls_model, cls_cols = load_models()


# ----------------------------
# Helper: Sanitize SQL input
# ----------------------------

def sanitize_sql(sql):
    """Strip whitespace and trailing semicolons that break EXPLAIN QUERY PLAN."""
    sql = sql.strip()
    sql = sql.rstrip(";").strip()
    return sql


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

    # Sanitize inputs
    a_sql = sanitize_sql(a_sql)
    b_sql = sanitize_sql(b_sql)

    if not a_sql or not b_sql:
        raise ValueError("SQL queries cannot be empty after sanitization.")

    db = sqlite3.connect(DB_PATH)

    try:
        # Query plans
        a_plan = explain(db, a_sql)
        b_plan = explain(db, b_sql)
    except sqlite3.OperationalError as e:
        db.close()
        raise ValueError(f"Invalid SQL query: {e}")
    finally:
        db.close()

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

    return a_plan, b_plan, a_lat, b_lat, winner, fa, fb


# ----------------------------
# Helper: Generate explanation
# ----------------------------

def generate_explanation(a_plan, b_plan, a_lat, b_lat, fa, fb, winner):
    """Analyze plan and feature differences to explain why one query is faster."""
    faster = "A" if winner == 1 else "B"
    slower = "B" if winner == 1 else "A"
    f_fast = fa if winner == 1 else fb
    f_slow = fb if winner == 1 else fa
    plan_fast = a_plan if winner == 1 else b_plan
    plan_slow = b_plan if winner == 1 else a_plan

    reasons = []

    # --- Plan-based reasons ---

    # Index usage
    fast_idx = "using index" in plan_fast.lower()
    slow_idx = "using index" in plan_slow.lower()
    if fast_idx and not slow_idx:
        reasons.append(f"📌 Query {faster} uses an **index lookup** while Query {slower} does not, avoiding expensive full-table scans.")
    elif fast_idx and slow_idx:
        fast_search = plan_fast.lower().count("search")
        slow_search = plan_slow.lower().count("search")
        if fast_search > slow_search:
            reasons.append(f"📌 Query {faster} uses **more index searches** ({fast_search} vs {slow_search}), enabling the database to locate rows faster.")

    # Full table scans
    fast_scans = plan_fast.lower().count("scan")
    slow_scans = plan_slow.lower().count("scan")
    if slow_scans > fast_scans:
        reasons.append(f"🔍 Query {slower} requires **{slow_scans} full table scan(s)** vs {fast_scans} for Query {faster}. Full scans read every row and are much slower on large tables.")

    # Correlated subqueries
    fast_corr = plan_fast.lower().count("correlated")
    slow_corr = plan_slow.lower().count("correlated")
    if slow_corr > fast_corr:
        reasons.append(f"🔄 Query {slower} uses **{slow_corr} correlated subquery/subqueries** (vs {fast_corr}). Correlated subqueries re-execute for every row in the outer query, which is expensive.")

    # Temp B-tree (sorting without index)
    if f_slow.get("plan_temp_btree", 0) > f_fast.get("plan_temp_btree", 0):
        reasons.append(f"🌡️ Query {slower} requires a **temporary B-tree** for sorting, meaning the ORDER BY column is not indexed.")

    # --- SQL structure reasons ---

    # JOIN vs subquery
    if f_fast.get("kw_join", 0) > f_slow.get("kw_join", 0) and f_slow.get("num_parens", 0) > f_fast.get("num_parens", 0):
        reasons.append(f"⚡ Query {faster} uses a **JOIN** while Query {slower} relies on **subqueries**. JOINs let the optimizer pick the best execution strategy, while nested subqueries often force row-by-row evaluation.")

    # EXISTS vs IN
    if f_fast.get("kw_exists", 0) > f_slow.get("kw_exists", 0) and f_slow.get("kw_in", 0) > f_fast.get("kw_in", 0):
        reasons.append(f"⚡ Query {faster} uses **EXISTS** while Query {slower} uses **IN**. EXISTS can short-circuit (stop early) once a match is found, while IN must build the full result set first.")
    elif f_fast.get("kw_in", 0) > f_slow.get("kw_in", 0) and f_slow.get("kw_exists", 0) > f_fast.get("kw_exists", 0):
        reasons.append(f"⚡ Query {faster} uses **IN** with a Bloom filter optimization, while Query {slower} uses **EXISTS** with a correlated lookup. The optimizer chose a more efficient strategy for IN in this case.")

    # Query complexity
    len_diff = f_slow.get("len_chars", 0) - f_fast.get("len_chars", 0)
    if len_diff > 30:
        reasons.append(f"📏 Query {slower} is **{len_diff} characters longer**, indicating higher structural complexity which often correlates with more work for the database engine.")

    # More predicates
    and_diff = f_slow.get("num_and", 0) - f_fast.get("num_and", 0)
    if and_diff > 0:
        reasons.append(f"🔗 Query {slower} has **{and_diff} more AND predicate(s)**, requiring additional filtering operations.")

    # Latency comparison
    lat_diff = abs(a_lat - b_lat)
    if lat_diff > 0.5:
        reasons.append(f"📊 The predicted latency gap is **{lat_diff:.2f}** (log-scale) — a **significant** performance difference.")
    elif lat_diff > 0.1:
        reasons.append(f"📊 The predicted latency gap is **{lat_diff:.2f}** (log-scale) — a **moderate** performance difference.")
    else:
        reasons.append(f"📊 The predicted latency gap is **{lat_diff:.2f}** (log-scale) — a **small** difference; both queries perform similarly.")

    # Fallback if no structural reasons found
    if len(reasons) <= 1:
        reasons.insert(0, f"🤖 The ML model detected subtle feature differences in SQL structure and execution plans that favor Query {faster}.")

    return reasons


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("🔍 QueryComparatorAI")
st.write("Compare two SQL queries and predict which one is faster using ML.")

sql_a = st.text_area("SQL Query A", height=160)
sql_b = st.text_area("SQL Query B", height=160)

if st.button("Compare Queries"):

    if not sql_a.strip() or not sql_b.strip():
        st.error("Please enter both SQL queries.")
    else:
        try:
            with st.spinner("Analyzing..."):
                a_plan, b_plan, a_lat, b_lat, winner, fa, fb = predict_pair(sql_a, sql_b)

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

            # Explanation section
            st.subheader("🧠 Why?")
            reasons = generate_explanation(a_plan, b_plan, a_lat, b_lat, fa, fb, winner)
            for reason in reasons:
                st.markdown(f"- {reason}")

        except ValueError as e:
            st.error(f"⚠️ {e}")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")