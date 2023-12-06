# RAG (Retrieval Augmented Generation) oraz wpływ różnych embeddingów na działanie tej metody.
Ten projekt bada wpływ różnych embeddingów na działanie metody Retrieval Augmented Generation (RAG) na przykładzie bazy wektorowej Chroma.

Główne cele projektu:
1. Analiza podejścia Retrieval Augmented Generation.
2. Zdefiniowanie testowanych embeddingów.
3. Zdefiniowanie metod ewaluacji skuteczności działania metody RAG.
4. Badanie wpływu różnych embeddingów na działanie metody RAG.
5. Wnioski.


## Czym jest RAG?
RAG jest jednym z najbardziej komercjalizowanych obszarów zastosowań generatywnej sztucznej inteligencji. Jest to skrót od "Retrieval-Augmented Generation," czyli w tłumaczeniu na polski "Generacja z użyciem mechanizmu odzyskiwania." RAG to podejście, które łączy w sobie dwa główne elementy: mechanizm odzyskiwania informacji (retrieval) i generację tekstu.

RAG jest często wykorzystywany w zadaniach związanych z przetwarzaniem języka naturalnego, takich jak systemy odpowiadające na pytania, generatory tekstu czy też w systemach dialogowych. Integracja mechanizmu odzyskiwania informacji ma na celu poprawę jakości generowanego tekstu poprzez korzystanie z istniejących, precyzyjnych danych.

Podejście RAG ma zastosowanie w różnych obszarach, takich jak wyszukiwarki internetowe, systemy wspomagające decyzje, czy aplikacje do automatycznego generowania treści. Wdrażanie takiego modelu może przyczynić się do lepszej skuteczności i dostarczenia bardziej trafnych odpowiedzi w kontekście danego zadania.

### Jak działa RAG?
![RAG [6]](assets/1695787886133.png)

RAG łączy procesy odzyskiwania informacji (Retrieval) i generacji tekstu (Generation). Działanie RAG można podzielić na kilka kluczowych etapów:

0. **Utworzenie wektorowej bazy danych**:
    - Podstawą jest zapewnienie wektorowej bazy danych zawierającej embeddingi naszej bazy wiedzy.

1. **Zapytanie Użytkownika**:
    - Wszystko rozpoczyna się od zapytania użytkownika, które jest przekazywane do systemu RAG. Zapytanie to może być pytaniem, prośbą o wygenerowanie tekstu lub jakimkolwiek innym promptem.

2. **Odzyskiwanie Informacji (Retrieval)**: 
    - Zapytanie użytkownika zostaje zembeddowane do bazy wektorowej. W tym kroku należy pamiętać aby wykorzystać ten sam model, którego wcześniej użyliśmy do utworzenia wektorowej bazy danych.
    - System pobiera *k* najbardziej podobnych elementów z bazy wektorowej. Do znalezienia najbardziej podobnych elementów pod względem semantycznym wykorzystywana jest odległość cosinusowa (cosine similarity).

3. **Generacja Tekstu (Generation)**:
    - Pobrane *k* najbardziej podobnych elementów jest łączone z zapytaniem użytkownika.
	- Tak przygotowane zapytanie użytkownika wraz z pobranym kontekstem jest wykorzystywane przez model generacji tekstu, aby stworzyć nową, spójną odpowiedź.
	- W tym projekcie do generacji tekstu wykorzystywany jest model *google/flan-ul2*.
	- Ta odpowiedź jest następnie przekazywana użytkownikowi jako końcowy produkt działania systemu RAG.
## Jak można ewaluować działanie systemu RAG?
Metodę RAG można ewaluować względem następujących aspektów:
- ewaluacja odzyskiwania informacji (tylko Retrieval)
    - Context Precision (dokładność kontekstu)
    - Context Recall (pełność kontekstu)
- ewaluacja generacji tekstu (tylko Generation)
    - Generation Faithfullness (wierność generacji)
    - Generation Relevance (znaczenie generacji)

![RAG - ewaluacja [5]](assets/1*4O0q0SV8d1SpcbkIpZbEIw.webp)

Chcąc ewaluować RAG całościowo, możemy patrzeć na średnią harmoniczną powyższych czterech metryk. W taki sposób działa metoda **ragas score** [4]. W tym projekcie skupimy się na ewaluacji odzyskiwania informacji (tylko Retrieval). Oczywiście zarówno aspekt Retrieval jak i Generation są niezwykle istotne, natomiast w naszym przypadku, tj. w przypadku badania wpływu różnych embeddingów szczególnie interesującym będzie właśnie kontekst Retrieval.

Zarówno Generation Faithfullness (czyli mierzenie halucynacji) jak i Generation Relevance (czyli mierzenie jak dobrze wygenerowana odpowiedź odnosi się do zadanego pytania) zależą przede wszystkim od modelu wykorzystanego do wygenerowania ostatecznej odpowiedzi. Natomiast model tworzący embeddingi będzie miał bezpośredni wpływ na aspekt odzyskiwania informacji (Retrieval). W związku z powyższym wybierając metodę ewaluacji skupiliśmy się tylko i wyłącznie na aspekcie Retrieval.


### Wybrana metoda ewaluacji
Dokonujemy badania miarą główną n-DCG@10 z miarami pomocniczymi MRR@k oraz MAP@k na zadaniu Retrieval projektu MTEB [2] na w sumie 15 benchmarkach po raz pierwszy skompilowanych na projekcie BEIR [3]. Do ewaluacji wykorzystywane jest narzędzie mteb, do podglądu rezultatów oraz oceny manualnej używany jest system RAG z wektorową bazą danych Chroma zbudowany w technologii LangChain z interfejsem w Streamlit.

- n-DCG@10 (Normalized Discounted Cumulative Gain at 10)
    - Skupia się na ocenie jakości pierwszych 10 wyników, co jest istotne w kontekście szybkiego dostarczania trafnych odpowiedzi użytkownikom.
- MRR@k (Mean Reciprocal Rank at k)
    - Ocenia, jak dobrze system odzyskuje najbardziej trafne informacje.
- MAP@k (Mean Average Precision at k)
    - Ocenia średnią dokładność w kontekście wielu zapytań, co jest ważne w ewaluacji skuteczności odzyskiwania informacji.

## Testowane embeddingi
- **BERT (Bidirectional Encoder Representations from Transformers)**: 
    - BERT jest jednym z najbardziej znanych modeli do generowania embeddingów. Jest on szczególnie efektywny w rozumieniu kontekstu, co może być kluczowe w metodzie RAG.
    - Na przykład: `bert-base-uncased`, `bert-large-uncased`.

- **GPT-2/GPT-3 (Generative Pretrained Transformer 2/3)**:
    - Te modele, znane z generowania spójnego i kontekstowego tekstu, mogą być użyteczne do badania, jak generatywne embeddingi wpływają na proces generacji w RAG.
    - Na przykład: `gpt2`, `EleutherAI/gpt-neo-2.7B`.

- **RoBERTa (A Robustly Optimized BERT Pretraining Approach)**:
    - RoBERTa jest wariantem BERT, który został zoptymalizowany pod kątem większej dokładności. Może to zapewnić interesujące porównanie z tradycyjnym BERT-em.
    - Na przykład: `roberta-base`, `roberta-large`.

- **DistilBERT (Distilled Version of BERT)**:
    - Jest to uproszczona i bardziej efektywna wersja BERT-a pod względem obliczeniowym, co może być istotne w zastosowaniach, gdzie szybkość jest kluczowa.
    - Na przykład: `distilbert-base-uncased`.

- **T5 (Text-To-Text Transfer Transformer)**:
    - T5 jest wszechstronnym modelem, który traktuje każde zadanie NLP jako zadanie konwersji tekstu na tekst, co może być interesujące w kontekście RAG.
    - Na przykład: `t5-small`, `t5-base`.

- **ALBERT (A Lite BERT)**:
   - ALBERT to uproszczona wersja BERT-a, która została opracowana przez Google w celu radzenia sobie z problemami wynikającymi z dużych rozmiarów modelu. Używa ona technik redukcji parametrów, co może być korzystne w kontekście efektywności obliczeniowej i skalowalności systemu RAG.
   - Na przykład: `albert-base-v2`, `albert-large-v2`.

## Wyniki eksperymentów
- [ ] TODO w dalszych etapach projektu

## Wnioski
- [ ] TODO w dalszych etapach projektu

## Źródła:
[1] Andrew Rosenberg and Julia Hirschberg. 2007. Vmeasure: A conditional entropy-based external cluster evaluation measure. pages 410–420.
[2] https://arxiv.org/abs/2210.07316
[3] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models
[4] https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
[5] https://cobusgreyling.medium.com/rag-evaluation-9813a931b3d4
[6] https://www.linkedin.com/pulse/what-retrieval-augmented-generation-grow-right/
