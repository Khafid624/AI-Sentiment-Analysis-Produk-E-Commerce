def predict_sentiment(text):

    # cek null
    if text is None:
        return "Netral"

    text = str(text)

    # berbagai bentuk kosong
    if text.strip().lower() in ["", "nan", "none", "null"]:
        return "Netral"

    text = clean_text(text)

    # setelah preprocessing
    if text.strip() == "":
        return "Netral"

    vector = tfidf.transform([text])

    prediction = model.predict(vector)[0]

    return prediction