import os
from flask import Flask, request
import telegram

app = Flask(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TOKEN)

@app.route('/')
def home():
    return "Mp ORBbot is alive!"

@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    if update.message and update.message.text == "/status":
        bot.send_message(chat_id=update.message.chat_id, text="ORBbot är igång ✅")
    return "ok"

if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=PORT)