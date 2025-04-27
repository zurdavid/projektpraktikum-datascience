# Datenlage

4 Trainings-Dateien:
- `products.csv`
- `stores.csv`
- `transaction_lines_train_3.parquet`
- `transactions_train_3.parquet` (2 mögliche Zielvariablen - label und damage)

Ausserdem 2 Dateien, mit Daten auf denen das Modell evaluiert werden soll:
- `transaction_lines_test_3.parquet`
- `transactions_test_3.parquet` (Zielvariablen fehlen)

## Datenqualität

Die Datenqualität ist auf den ersten Blick gut. Es gibt keine Spalten die sofort verworfen werden sollten.

## Join

In einem ersten Schritt soll aus den 4 Tabellen eine Join-Tabelle erzeugt werden.

Je nach Join-Strategie können unbeabsichtigt Zeilen verloren gehen, da kein match möglich war, Duplikate oder Zeilen mit fehlenden in vielen Spalten enstehen.

Daher ist es sinnvoll im Voraus zu prüfen, inwiefern die Tabellen zusammenpassen um mögliche Probleme noch vor dem Join zu erkennen.

### Join Transactions und Stores

- siehe `analyse_join_transactions_and_stores.ipynb`
- jede Transaktion hat eine gültige Store-ID
- die Transaktionen stammen aus 5 verschidenen Filialen
- join ist unproblematisch

### Join Transactions und Lines

- siehe `analyse_join_transactions_and_lines.ipynb`
- 13 Transaktionen haben keine zugehörigen Lines (konsistent mit der Angaben *n_lines* = 0 in den Transaktionen)
- es gibt aber auch Transaktionen mit *n_lines* = 0 und zugehörigen Lines (*was_voided* ist dann jeweils *true*)
- nur eine dieser Transaktionen ist gelabelt
- es ist sinnvoll, diese Transaktionen nach dem Join zu entfernen

### Join Lines und Products

- siehe `analyse_join_lines_and_products.ipynb`
- bei 192 Lines fehlt die product_id
- Auffälligkeiten in diesen Zeilen:
    - *was_voided* = true
    - Kamera hat die Produkte nicht erkannt (mit hoher Sicherheit)
    - in 16 Fällen gehören die Lines zu einer gelabelten Transaktion
    - das Label ist in allen Fällen "FRAUD"
- die Zeilen sollten nicht entfernt werden, da sie statistisch relevant sind und in solchen Fällen immer eine Kontrolle stattfinden sollte
- offene Fragen:
    - was könnte der Grund für die fehlenden product_ids sein?
    - wie sollen die fehlenden Werte aus der Tabelle Products für diese Zeilen behandelt werden?

## Missing Values

### Spalte "customer_satisfaction"
- siehe `analyse_missing_values.ipynb`
- Die Spalte customer_feedback enthält nur in 7.6% der Fälle einen Wert. Der Mittelwert ist mit 9.3 aussergewöhnlich hoch, schon das 25%Quantil liegt bei 10.0. Es ist daher fraglich ob die Spalte für die Analyse nützlich ist. (weitere Analysen wären vielleicht sinnvoll)
- Das Vorhandensein eines Wertes in der Spalte scheint aber statistisch relevant zu sein: ein Chi-Quadrat-Test ergibt einen p-Wert von 0.0011
- Die Spalte könnte durch eine binäre Spalte "has_customer_feedback" ersetzt werden
