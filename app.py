import streamlit as st
import pickle

# Page Title
st.title("Fake News Detection")

try:
    # Load model and vectorizer
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    # User input
    news = st.text_area("Enter News Article")

    # Predict button
    if st.button("Predict"):

        # Check empty input
        if news.strip() == "":
            st.warning("Please enter news text.")

        else:
            # Transform input text
            transformed_news = vectorizer.transform([news])

            # Predict
            prediction = model.predict(transformed_news)

            # Get raw prediction value
            raw_prediction = prediction[0]

            # Convert to string for display
            prediction_text = str(raw_prediction).lower().strip()

            # Display prediction value
            st.write("Prediction Value:", prediction_text)

            # ── Label mapping ────────────────────────────────────────────────
            # The model may output either:
            #   (A) Numeric binary labels  →  0 = FAKE, 1 = REAL
            #   (B) Raw LIAR string labels →  6-class labels from the dataset
            #
            # LIAR dataset 6 classes:
            #   FAKE: "false", "pants-fire", "barely-true", "half-true"
            #   REAL: "true", "mostly-true"

            fake_labels = {"false", "pants-fire", "barely-true", "half-true", "0"}
            real_labels = {"true", "mostly-true", "1"}

            if prediction_text in fake_labels:
                st.error("🚨 Fake News")
            elif prediction_text in real_labels:
                st.success("✅ Real News")
            else:
                # Fallback: unknown label — show it and warn
                st.warning(
                    f"⚠️ Unrecognised label: '{prediction_text}'. "
                    "Check that your model.pkl was trained on the LIAR dataset."
                )

except FileNotFoundError as e:
    st.error(
        f"Model file not found: {e}\n\n"
        "Make sure **model.pkl** and **vectorizer.pkl** are in the same "
        "folder as app.py."
    )
except Exception as e:
    st.error(f"Error: {e}")
