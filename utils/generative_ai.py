from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_insight(df):

    summary_text = df["sentiment"].value_counts().to_string()

    prompt = f"""
    Kamu adalah data analyst.

    Berikut hasil analisis sentiment:
    {summary_text}

    Berikan:
    1. Insight bisnis
    2. Kesimpulan
    3. Rekomendasi strategi
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content