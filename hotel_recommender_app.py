import tkinter as tk
from tkinter import scrolledtext, messagebox
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка модели и данных
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df_grouped = pd.read_csv("Data/grouped_hotels.csv")

# Функция поиска отелей
def recommend_hotels():
    user_input = entry.get()
    
    if not user_input.strip():
        messagebox.showwarning("Ошибка", "Введите предпочтения перед поиском!")
        return
    
    user_vector = vectorizer.transform([user_input])  # Преобразуем ввод в вектор
    hotel_vectors = vectorizer.transform(df_grouped["clean_text"])  # Векторы отелей

    similarities = cosine_similarity(user_vector, hotel_vectors).flatten()  # Считаем сходство
    top_indices = similarities.argsort()[::-1][:5]  # Берём топ-5

    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)

    if similarities[top_indices[0]] == 0:
        result_text.insert(tk.END, "❌ По вашему запросу ничего не найдено.")
    else:
        result_text.insert(tk.END, "🎯 Топ 5 рекомендаций:\n\n")
        for idx, i in enumerate(top_indices, 1):
            result_text.insert(
                tk.END, f"{idx}. {df_grouped['name_ru'][i]} (Рейтинг: {df_grouped['rating'][i]:.2f})\n"
            )
    
    result_text.config(state=tk.DISABLED)

# Создание GUI
app = tk.Tk()
app.title("Рекомендации отелей")
app.geometry("500x400")

# Виджеты
tk.Label(app, text="Введите предпочтения:", font=("Arial", 12)).pack(pady=5)
entry = tk.Entry(app, width=50, font=("Arial", 12))
entry.pack(pady=5)

tk.Button(app, text="🔍 Найти отели", font=("Arial", 12), command=recommend_hotels).pack(pady=10)

result_text = scrolledtext.ScrolledText(app, width=60, height=10, font=("Arial", 10), state=tk.DISABLED)
result_text.pack(pady=5)

# Запуск приложения
app.mainloop()
