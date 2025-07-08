ai_mode = "neutral"

def set_ai_mode(mode):
    global ai_mode
    if mode in ["neutral", "aggressiv", "försiktig"]:
        ai_mode = mode
        return f"✅ AI-läge satt till: {mode}"
    return "❌ Ogiltigt AI-läge. Välj mellan: neutral, aggressiv, försiktig."

def get_ai_mode():
    return ai_mode