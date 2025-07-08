import csv

def export_csv():
    filename = "/mnt/data/mp_orbbot_result.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Datum", "TradeTyp", "Pris", "Resultat"])
        writer.writerow(["BTCUSDT", "2025-07-08", "Buy", "58200", "+1.2%"])
    return filename