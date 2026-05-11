# Evaluation Roadmap

Stand und nächste Schritte für die RAGAS-Evaluation. Dient als Orientierung
zwischen Sessions: woran wir gerade arbeiten, was als nächstes kommt, warum.

---

## Status quo

**Was steht:**
- Zwei-Ebenen-Eval: referenzfrei (`evaluate.py`) + Ground Truth (`evaluate_ground_truth.py`).
- Externe GT aus BpB Wahl-O-Mat 2025 (38 Thesen × 7 Parteien = 266 Einträge).
- Stratified Sampling pro Partei (`--sample N`).
- 4 dokumentierte Varianten-Läufe (k=5/10, Section-aware, e5-large).
- README mit explizit benannten methodischen Limitationen.

**Was fehlt methodisch:**
- Konfidenzintervalle → Score-Unterschiede sind aktuell nicht als signifikant belegbar.
- Judge-Noise nicht quantifiziert → unbekannter Rausch-Floor.
- Per-Row-Scores werden nicht persistiert → keine Fehleranalyse möglich.
- Stance Detection (zentrales Produkt-Feature) wird gar nicht evaluiert.
- Frageformulierung verwirft den These-Text → künstlich breite Frage vs. spezifische GT.

---

## Nächste Schritte (priorisiert)

Reihenfolge so gewählt, dass jeder Schritt eine im README benannte Limitation
ersetzt. Jeder Schritt ist allein <1h Arbeit; (a) und (c) gehören technisch zusammen
und sollten in einer Session gemacht werden.

### (a) Bootstrap-CIs auf Per-Row-Scores
**Warum:** Aktuell stehen Punkt-Schätzungen in der Vergleichstabelle ohne Streumaß.
Damit lässt sich nicht sagen, ob 0.621 vs. 0.625 ein echter Unterschied ist.
CIs ersetzen Bauchgefühl durch eine harte Aussage über Überlappung.

**Wie:** `scipy.stats.bootstrap` über `result.to_pandas()`-Spalten, n_resamples=10000,
CI 95%. Berichten als `0.621 [0.55, 0.69]` in der README-Tabelle.

**Was es ersetzt:** Limitation "n=35 ist statistisch dünn; Bootstrap-CIs fehlen".

### (c) Per-Row-Scores als CSV persistieren
**Warum:** Voraussetzung für (a), (b) und jede Fehleranalyse. Aktuell wird nur
der Mittelwert in JSON gespeichert; die Rohdaten gehen verloren.

**Wie:** `df.to_csv(RESULTS_DIR / f"raw_{run_id}.csv")` neben dem aggregierten JSON.
`run_id` = Timestamp + Variantenname. Spalten: question, party, alle Metrik-Scores.

**Was es ersetzt:** Implizit – macht Reproduzierbarkeit überhaupt erst möglich.

### (b) Judge-Noise einmal messen
**Warum:** Alle fünf RAGAS-Metriken sind LLM-as-Judge. Ohne bekannten Rausch-Floor
weiß man nicht, ab welchem Delta ein Unterschied diskussionsfähig ist.

**Wie:** Baseline 3× hintereinander laufen lassen (identische Pipeline, nur RAGAS-Judge
neu). Stdev der Mittelwerte als "Judge-Floor" notieren. Effekte < 2×Stdev gelten
fortan als nicht belegt.

**Was es ersetzt:** Limitation "LLM-Judge-Varianz wurde nicht durch Mehrfachläufe gemessen".

### (d) Stance-Accuracy als kategoriale Metrik
**Warum:** Größte Eval-Lücke. Stance Detection ist ein Kern-Feature der App,
wird aber nirgends gemessen. Außerdem: keine LLM-Judge nötig → kein Rauschen,
kein Längen-Bias, schärfster Score im ganzen Eval.

**Wie:** Neues Skript `evaluation/evaluate_stance.py`. Für jeden GT-Eintrag den
`/compare`-Endpoint (oder direkt `run_compare`) aufrufen, Stance aus der Antwort
extrahieren, gegen `position` aus der GT matchen. Accuracy + Confusion Matrix
+ Per-Party-Breakdown.

**Was es ersetzt:** Limitation "Stance-Accuracy fehlt bisher".

### (e) Frageformulierungs-Variante
**Warum:** Aktuell fragt der Build-Schritt generisch
`"Was ist die Position der {Partei} zum Thema {Titel}?"` — die spezifische These
geht verloren. Das ist eine plausible Erklärung für das ~0.62-Niveau bei
context_recall. Eine Vergleichsvariante würde das empirisch klären.

**Wie:** In `build_dataset.py` zweite Frageform ergänzen:
`"Wie steht die {Partei} zu folgender These: '{These-Text}'?"`. Baseline-Run
einmal mit beiden Varianten, ctx_recall vergleichen.

**Was es ersetzt:** Limitation "Frageformulierung verwirft den These-Text".

---

## Was wir bewusst zurückstellen

- **HyDE-Ablation:** wichtig, aber wenn (a)+(b) zeigen, dass selbst große Unterschiede
  im Judge-Rauschen liegen, ist die Ablation methodisch erst danach interpretierbar.
- **Orthogonales Varianten-Gitter** (Section-aware × k=10 etc.): nur sinnvoll, sobald
  CIs existieren und zeigen, dass Effekte überhaupt diskriminierbar sind.
- **Adversariales Testset für Eval 1:** für Demo-Showcase nice-to-have, für
  methodische Sauberkeit nicht kritisch.

---

## Pfad zu Production-Reife (Skizze, nicht für Demo)

Eval-Mindset ändert sich von "einmaliger Messung" zu "kontinuierlichem Signal":

1. **Regressions-Eval als CI-Gate**
   Mini-Set (10–20 Einträge) läuft bei jedem Push. Stance-Accuracy + Faithfulness
   über Threshold = grün. Vollrun nur bei Release-Tag.

2. **Offline-Eval vs. Online-Eval trennen**
   Produktiver Traffic loggt `(question, chunks, answer)`. Wöchentlicher Sample-Run
   mit Faithfulness/Relevancy → Drift-Detection.

3. **Human-Eval-Loop**
   Antworten mit `faithfulness < 0.5` werden geflaggt. Wöchentliches Review-Sample
   (~20) wird gelabelt → wächst in den Ground-Truth-Datensatz hinein.

Für die Demo reicht es, Schritt 1 zu skizzieren: ein Mini-Set + GitHub Action,
die Scores in die PR-Description postet. Visuell stark, konzeptionell zeigt es,
dass Eval als kontinuierlicher Prozess gedacht wird.

---

## Aktuelle Session-Notiz

_Hier eintragen, woran zuletzt gearbeitet wurde, damit die nächste Session
direkt einsteigen kann._

- [ ] (a) Bootstrap-CIs
- [ ] (c) Per-Row-CSV-Dump
- [ ] (b) Judge-Noise-Messung (3 Wiederholläufe Baseline)
- [ ] (d) Stance-Accuracy-Skript
- [ ] (e) Frageformulierungs-Variante
