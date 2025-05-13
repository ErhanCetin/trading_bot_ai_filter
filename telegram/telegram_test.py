from telegram.telegram_notifier import send_telegram_message

msg = (
    "*🚨 Sinyal Bildirimi*\n\n"
    "*BTCUSDT* 🔥 `LONG`\n"
    "`Entry:` 62700\n"
    "`SL:` 61800\n"
    "`TP:` 64000\n\n"
    "*Güç Skoru:* 4 / 4"
)

send_telegram_message(msg)
