import pandas as pd

# 1. Die kleine Frequenz-Datei einlesen (ca. 37 MB)
print("Lese usage_frequency.csv ein...")
df_freq = pd.read_csv('../../data/usage_frequency.csv')

# 2. Aggregieren: Alle 'pickup_counts' pro 'station_name' über alle Tage aufsummieren
print("Berechne die Gesamtanzahl der Ausleihen pro Station...")
station_totals = df_freq.groupby('station_name')['pickup_counts'].sum().reset_index()

# 3. Sortieren: Von der Station mit den meisten Ausleihen absteigend
station_totals = station_totals.sort_values(by='pickup_counts', ascending=False)

# 4. Die Top 50 Stationen auswählen
top_50_df = station_totals.head(50)

# 5. Nur die Namen der Stationen als Python-Liste speichern (für Schritt 2)
top_50_stations = top_50_df['station_name'].tolist()

# Kontrolle: Ausgabe der Top 10 Stationen und deren Gesamt-Mieten
print(f"\nDie Top 50 Stationen wurden erfolgreich extrahiert!")
print("\nHier sind die Top 10 zur Kontrolle:")
print(top_50_df.head(50))

# Optional: Speichern Sie diese Liste für Ihren Report (Deliverable 2)
# top_50_df.to_csv('top_50_stations.csv', index=False)