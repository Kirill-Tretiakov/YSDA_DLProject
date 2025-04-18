YSDA_DLProject. Лабораторная работа 4. HF tune Transformers

Ссылка на чат с ботом: https://t.me/+434m1LFHb0c2NzNi

# Классификатор сообщений в Telegram-чате на основе RuBERT

### Описание проекта

Telegram-бот, основанный на модифицированной модели RuBERT, анализирует сообщения в групповом чате и автоматически распределяет их по категориям:

 - Вопросы (/questions)
 - Отзывы (/feedback)
 - Важные сообщения (/important)
 - Спам (/spam – удаляется автоматически).

Бот работает в двух режимах:
 - Silent Mode: только категоризация без уведомлений, вывод информации по команде.
 - Demo Mode: категоризация с уведомления, бот определяет и отправляет категорию после каждого сообщения в чате.

### Техническая реализация

Базовая модель: [blanchefort/rubert-base-cased-sentiment](https://huggingface.co/blanchefort/rubert-base-cased-sentiment) (предобученный RuBERT для анализа тональности).

Дообучение: Замена последнего слоя на кастомную трехслойную нейросеть (CustomModel) и fine-tuning под задачу классификации чат-сообщений.

Датасет: Собранные вручную примеры сообщений (dataset.json).

Код обучения в ноутбуке: model_train.ipynb.

### Запуск бота

1. Настройка:

```python
telegram_token = "ВАШ_ТОКЕН"
model_path = "./model"
silent_mode = False
```

2. Установка зависимостей:

```bash
pip install requirements.txt
```

3. Запуск:

```bash
python bot.py
```

