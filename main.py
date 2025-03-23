import pandas as pd
import re
import nltk
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib
from tqdm import tqdm
from scipy.sparse import vstack
import numpy as np

# Загрузка данных из CSV-файла
df = pd.read_csv('Data/geo-reviews-dataset-2023.csv', encoding='utf-8')
df = df.loc[df['rubrics'] == 'Гостиница']  # Оставляем только гостиницы

# Инициализация инструментов для обработки текста
nltk.download('stopwords')
stanza.download('ru')
nlp = stanza.Pipeline('ru', processors='tokenize,lemma')

# Определение стоп-слов для удаления из текста
stop_words = set(nltk.corpus.stopwords.words('russian'))
additional_stops = {'что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', 'руб', 'мой', 'твой', 'его', 'её', 'наш', 'ваш', 'их', 'свой', 'еще', 'очень', 'поэтому', 'однако', 'конечно'}
stop_words.update(additional_stops)

# Функция предобработки текста (приведение к нижнему регистру и удаление лишних символов)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    return text

# Применение предобработки к отзывам
df['clean_text'] = df['text'].astype(str).apply(preprocess_text)

# Группировка отзывов по отелям и объединение их в один текст
df_grouped = df.groupby('name_ru')['clean_text'].apply(lambda x: ' '.join(x)).reset_index()
df_grouped['rating'] = df.groupby('name_ru')['rating'].mean().reset_index(drop=True)  # Средний рейтинг отеля

# Сохранение сгруппированных данных в CSV
df_grouped.to_csv('Data/grouped_hotels.csv', index=False, encoding='utf-8')

# Векторизация текстов отзывов с использованием TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stop_words))
X = vectorizer.fit_transform(df_grouped['clean_text'])
y = df_grouped['rating'].values

# Инициализация модели случайного леса для регрессии
model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
batch_size = 100  # Размер батча для обучения

# Обучение модели по батчам
for i in tqdm(range(0, X.shape[0], batch_size), desc="Обучение модели по батчам"):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    if i == 0:
        model.fit(X_batch, y_batch)  # Первоначальное обучение модели
    else:
        model.n_estimators += 10  # Увеличение количества деревьев в лесу
        model.fit(X_batch, y_batch)

# Сохранение обученной модели и векторайзера в файлы
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')