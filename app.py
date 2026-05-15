import streamlit as st
import pickle
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detection", page_icon="🔍")
st.title("Fake News Detection")

# ── Load model & vectorizer (cached so they load only once) ───────────────────
@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_model()
except FileNotFoundError as e:
    st.error(
        f"❌ Model file not found: {e}\n\n"
        "Make sure **model.pkl** and **vectorizer.pkl** are in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── User input ────────────────────────────────────────────────────────────────
news = st.text_area("Enter News Article", height=150)

if st.button("Predict"):

    # ── Guard: empty input ────────────────────────────────────────────────────
    if not news.strip():
        st.warning("⚠️ Please enter some news text before predicting.")
        st.stop()

    try:
        # ── Transform & predict ───────────────────────────────────────────────
        transformed = vectorizer.transform([news])
        raw_pred    = model.predict(transformed)[0]

        # Normalise to string for display and matching
        prediction_text = str(raw_pred).lower().strip()
        st.write("Prediction Value:", prediction_text)

        # ── Label mapping ─────────────────────────────────────────────────────
        #
        # Your model may output ONE of two formats depending on how it was trained:
        #
        # (A) Binary numeric  →  0 = FAKE,  1 = REAL
        # (B) LIAR 6-class strings:
        #       FAKE  →  "false", "pants-fire", "barely-true", "half-true"
        #       REAL  →  "true",  "mostly-true"
        #
        # "half-true" is classified as FAKE because it still contains
        # significant misinformation — only "true" and "mostly-true"
        # are considered reliably accurate.

        FAKE_LABELS = {"false", "pants-fire", "barely-true", "half-true", "0"}
        REAL_LABELS = {"true", "mostly-true", "1"}

        if prediction_text in FAKE_LABELS:
            st.error("🚨 Fake News")

        elif prediction_text in REAL_LABELS:
            st.success("✅ Real News")

        else:
            # ── Fallback: numeric float (e.g. 0.0 / 1.0 from some models) ────
            try:
                numeric = float(prediction_text)
                if numeric < 0.5:
                    st.error("🚨 Fake News")
                else:
                    st.success("✅ Real News")
            except ValueError:
                st.warning(
                    f"⚠️ Unrecognised label: **'{prediction_text}'**\n\n"
                    "The model returned an unexpected output. "
                    "Check that **model.pkl** was trained on the LIAR dataset "
                    "and outputs one of: `false`, `pants-fire`, `barely-true`, "
                    "`half-true`, `true`, `mostly-true`, `0`, or `1`."
                )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
