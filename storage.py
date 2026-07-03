import json
import os

from config import CHAT_ID_FILE


# ==========================
# CHAT ID
# ==========================

def save_chat_id(chat_id):

    with open(CHAT_ID_FILE, "w") as f:

        f.write(str(chat_id))


def load_chat_id():

    if not os.path.exists(CHAT_ID_FILE):
        return None

    with open(CHAT_ID_FILE, "r") as f:

        return f.read().strip()


# ==========================
# SETTINGS
# ==========================

SETTINGS_FILE = "settings.json"


DEFAULT_SETTINGS = {

    "stake": 30,

    "max_open_trades": 5,

    "entry_mode": "retest",

    "trail_percent": 0.006,

    "breakout_buffer": 0.0003,

    "trend_filter": True,

    "volume_filter": True,

    "orb_hour": 7,

    "orb_minute": 0
}


def load_settings():

    if not os.path.exists(SETTINGS_FILE):

        save_settings(DEFAULT_SETTINGS)

        return DEFAULT_SETTINGS.copy()

    with open(SETTINGS_FILE, "r") as f:

        return json.load(f)


def save_settings(settings):

    with open(SETTINGS_FILE, "w") as f:

        json.dump(
            settings,
            f,
            indent=4
        )
