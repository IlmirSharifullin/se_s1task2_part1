from transformers import pipeline

sentiment_analyzer = pipeline(
    "text-classification",
    model="blanchefort/rubert-base-cased-sentiment",
    tokenizer="blanchefort/rubert-base-cased-sentiment"
)

if __name__ == '__main__':
    text = "Мне очень понравился этот фильм! Он был просто великолепен."

    result = sentiment_analyzer(text)

    print(f"Текст: {text}")
    print(f"Тональность: {result[0]['label']}")
    print(f"Уверенность: {result[0]['score']:.2f}")
