# RAG (Retrieval Augmented Generation) oraz wpływ różnych embeddingów na działanie tej metody.
Ten projekt bada wpływ różnych embeddingów na działanie metody Retrieval Augmented Generation (RAG) na przykładzie bazy wektorowej Chroma.

Główne cele projektu:
1. Analiza podejścia Retrieval Augmented Generation.
2. Zdefiniowanie testowanych embeddingów.
3. Zdefiniowanie metod ewaluacji skuteczności działania metody RAG.
4. Badanie wpływu różnych embeddingów na działanie metody RAG.
5. Wnioski.


## Czym jest RAG?
RAG jest jednym z najbardziej komercjalizowanych obszarów zastosowań generatywnej sztucznej inteligencji. Jest to skrót od "Retrieval-Augmented Generation," czyli w tłumaczeniu na polski "Generacja z użyciem mechanizmu odzyskiwania." RAG to podejście w dziedzinie sztucznej inteligencji (SI), które łączy w sobie dwa główne elementy: mechanizm odzyskiwania informacji (retrieval) i generację tekstu.

**Odzyskiwanie informacji (Retrieval)**: Ten aspekt odnosi się do zdolności systemu do pobierania istniejących informacji z dużych baz danych czy korpusów tekstowych. Może to być realizowane poprzez wykorzystanie zaawansowanych technik przeszukiwania, indeksowania lub modeli opartych na uczeniu maszynowym, które są w stanie identyfikować istotne informacje. W obecnym badaniu odzyskiwanie informacji będzie następowało poprzez odpytywanie bazy wektorowej o zadanym modelu embeddingu.

**Generacja tekstu (Generation)**: Drugi element dotyczy zdolności systemu do generowania nowego, spójnego tekstu na podstawie pobranych informacji. Może to obejmować odpowiedzi na pytania, tworzenie opisów, czy redagowanie tekstu na podstawie istniejących danych. W obecnym badaniu będzie to za każdym razem wykonywane przez ten sam model - *google/flan-ul2*.

RAG jest często wykorzystywany w zadaniach związanych z przetwarzaniem języka naturalnego, takich jak systemy odpowiadające na pytania, generatory tekstu czy też w systemach dialogowych. Integracja mechanizmu odzyskiwania informacji ma na celu poprawę jakości generowanego tekstu poprzez korzystanie z istniejących, precyzyjnych danych.

Podejście RAG ma zastosowanie w różnych obszarach, takich jak wyszukiwarki internetowe, systemy wspomagające decyzje, czy aplikacje do automatycznego generowania treści. Wdrażanie takiego modelu może przyczynić się do lepszej skuteczności i dostarczenia bardziej trafnych odpowiedzi w kontekście danego zadania.

## Testowane embeddingi
----coś tam coś tam, nazwy modeli z HF najlepiej-----


## Metody ewaluacji
Dokonujemy badania miarą główną n-DCG@10 z miarami pomocniczymi MRR@k oraz MAP@k na zadaniu Retrieval projektu MTEB [2] na w sumie 15 benchmarkach po raz pierwszy skompilowanych na projekcie BEIR [3]. Do ewaluacji wykorzystywane jest narzędzie mteb, do podglądu rezultatów oraz oceny manualnej używany jest system RAG z wektorową bazą danych Chroma zbudowany w technologii LangChain z interfejsem w Streamlit.

[1] Andrew Rosenberg and Julia Hirschberg. 2007. Vmeasure: A conditional entropy-based external cluster evaluation measure. pages 410–420.
[2] https://arxiv.org/abs/2210.07316
[3] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models

## Wyniki eksperymentów


## Wnioski