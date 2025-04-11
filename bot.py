import telebot
from config import BOT_TOKEN

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from model import CustomModel

class ChatAdministratorBot:
    def __init__(self, api_token, model_path, silent_mode=False):
        self.bot = telebot.TeleBot(api_token)
        self.tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = CustomModel.from_config(self.config)
        
        self.model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
        self.model.eval()

        self.silent_mode = silent_mode
        
        self.questions = []
        self.feedback = []
        self.important = []
        self.spam = []

        self.bot.message_handler(commands=['start'])(self.send_welcome)
        self.bot.message_handler(commands=['help'])(self.send_help)
        self.bot.message_handler(commands=['questions'])(self.send_questions)
        self.bot.message_handler(commands=['feedback'])(self.send_feedback)
        self.bot.message_handler(commands=['important'])(self.send_important)
        self.bot.message_handler(commands=['spam'])(self.send_spam)
        self.bot.message_handler(func=lambda message: True)(self.message_processing)

    def send_welcome(self, message):
        """Обработчик команды /start"""
        self.bot.reply_to(message, "Привет, я бот-администратор 😎. Умею удалять спам 🚫, собирать вместе вопросы 📝 и отзывы ⭐️")

    def send_help(self, message):
        """Обработчик команды /help"""
        self.bot.reply_to(
            message,
            "📚 Помощь по командам бота 📚\n\n"
            "Этот бот помогает организовывать и фильтровать сообщения в чате. Вот что он умеет:\n\n"
            "🔹 /questions – Показать накопленные вопросы 📝\n"
            "🔹 /feedback – Показать полученные отзывы ⭐️\n"
            "🔹 /important – Показать важные сообщения ❗️\n"
            "🔹 /spam – Показать удалённые сообщения со спамом 🚫"
        )

    def send_questions(self, message):
        """Обработчик команды /questions - показывает накопленные вопросы"""
        if self.questions:
            self.bot.send_message(
                message.chat.id,
                "Накопленные вопросы 📋:\n\n" + "\n".join(self.questions)
            )
            self.questions = []
        else:
            self.bot.send_message(message.chat.id, "Новых вопросов не было 😎")

    def send_feedback(self, message):
        """Обработчик команды /feedback - показывает накопленные отзывы"""
        if self.feedback:
            self.bot.send_message(
                message.chat.id,
                "Накопленные отзывы 📋:\n\n" + "\n".join(self.feedback)
            )
            self.feedback = []
        else:
            self.bot.send_message(message.chat.id, "Новых отзывов не было 😎")

    def send_important(self, message):
        """Обработчик команды /important - показывает накопленные важные сообщения"""
        if self.important:
            self.bot.send_message(
                message.chat.id,
                "Накопленные важные сообщения 📋:\n\n" + "\n".join(self.important)
            )
            self.important = []
        else:
            self.bot.send_message(message.chat.id, "Новых важных сообщений не было 😎")

    def send_spam(self, message):
        """Обработчик команды /spam - показывает удалённые по подозрению на спам сообщения"""
        if self.spam:
            self.bot.send_message(
                message.chat.id,
                "Удалённые сообщения, содержащие спам 📕:\n\n" + "\n".join(self.spam)
            )
            self.spam = []
        else:
            self.bot.send_message(message.chat.id, "Новых сообщений со спамом не было обнаружено 😎")

    def message_processing(self, message):
        """Обработчик всех сообщений"""

        id2label = self.model.config.id2label

        inputs = self.tokenizer(message.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        if id2label[predicted_class] == "QUESTION":
            self.questions.append(message.text)
        elif id2label[predicted_class] == "FEEDBACK":
            self.feedback.append(message.text)
        elif id2label[predicted_class] == "IMPORTANT":
            self.important.append(message.text)
        elif id2label[predicted_class] == "SPAM":
            self.spam.append(message.text)
            try:
                self.bot.delete_message(message.chat.id, message.message_id)
                self.bot.send_message(message.chat.id, "Сообщение было удалено 🚨")
            except Exception as e:
                print(f"Ошибка при удалении сообщения: {e}. Боту не выданы права на удаление сообщений.")

        if not self.silent_mode:
            self.bot.send_message(message.chat.id, id2label[predicted_class])

    def run(self):
        """Запускает бота"""
        print("Бот запущен...")
        self.bot.polling(none_stop=True)

if __name__ == "__main__":
    model_path = "./model"

    bot = ChatAdministratorBot(BOT_TOKEN, model_path)
    bot.run()
