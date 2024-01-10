Richard Staszkiewicz
Miłosz Łopatto

# Dokumentacja wstępna do projektu z przedmiotu NLP

Spis treści:
- [[#Temat projektu|Temat projektu]]
- [[#Cel projektu|Cel projektu]]
- [[#Czym jest RAG?|Czym jest RAG?]]
	- [[#Czym jest RAG?#Jak działa RAG?|Jak działa RAG?]]
- [[#Jak można ewaluować działanie systemu RAG?|Jak można ewaluować działanie systemu RAG?]]
	- [[#Jak można ewaluować działanie systemu RAG?#Wybrana metoda ewaluacji|Wybrana metoda ewaluacji]]
- [[#Przykładowe modele embeddingowe|Przykładowe modele embeddingowe]]
- [[#Wyniki eksperymentów|Wyniki eksperymentów]]
- [[#Wnioski|Wnioski]]
- [[#Źródła|Źródła]]


## Temat projektu

> [!info] Temat projektu
>**RAG (Retrieval Augmented Generation) oraz wpływ różnych embeddingów na działanie tej metody.**

Projekt ten bada wpływ różnych embeddingów na działanie metody Retrieval Augmented Generation (RAG).

## Cel projektu
Początkowo chcieliśmy porównać kilka bardziej znanych modeli generujących embeddingi. Jednakże po odkryciu [benchmarku MTEB (Massive Text Embedding Benchmark)](https://huggingface.co/spaces/mteb/leaderboard) [9] stwierdziliśmy, że lepszym pomysłem byłoby wybranie modelu, który nie został jeszcze uwzględniony w tym benchmarku. Dzięki temu zamiast liczyć coś co zostało już wcześniej policzone w jakimś stopniu przyczynimy się do rozwoju tego otwarto-źródłowego projektu, ponieważ uzyskane przez nas wyniki wgramy do wspomnianej wyżej tabeli wyników.

Zakres projektu:
1. Analiza podejścia Retrieval Augmented Generation.
2. Zdefiniowanie metod ewaluacji skuteczności działania metody RAG.
3. Zdefiniowanie testowanych embeddingów.
4. Przeprowadzenie testów.
5. Analiza wyników i wnioski.


## Czym jest RAG?
Jest to skrót od "Retrieval-Augmented Generation", czyli w tłumaczeniu na polski "Generacja z użyciem mechanizmu odzyskiwania". RAG jest często wykorzystywany w zadaniach związanych z przetwarzaniem języka naturalnego, takich jak systemy odpowiadające na pytania, generatory tekstu czy też w systemach dialogowych. Wdrażanie takiego systemu może przyczynić się do lepszej skuteczności i dostarczenia bardziej trafnych oraz aktualnych odpowiedzi w kontekście danego zadania. [7, 8]

### Jak działa RAG?
![RAG [6]](assets/rag.png)

RAG łączy procesy odzyskiwania informacji (Retrieval) i generacji tekstu (Generation). Działanie RAG można podzielić na kilka kluczowych etapów:

0. **Utworzenie bazy wiedzy**:
    - Podstawą jest utworzenie bazy wiedzy do przeszukiwania w kontekście zapytań użytkownika. W naszym projekcie jej realizacja będzie miała postać bazy wektorowej zawierającej embeddingi pochodzące z dokumentów PDF zapewnionych przez użytkownika.

1. **Zapytanie Użytkownika**:
    - Zapytanie użytkownika jest przekazywane do systemu RAG. Zapytanie to może być pytaniem, prośbą o wygenerowanie tekstu lub jakimkolwiek innym promptem.

2. **Odzyskiwanie Informacji (Retrieval)**:
    - Zapytanie użytkownika zostaje zembeddowane do bazy wektorowej. W tym kroku należy pamiętać aby wykorzystać ten sam model, którego wcześniej użyliśmy do utworzenia wektorowej bazy danych.
    - System pobiera *k* najbardziej podobnych elementów z bazy wektorowej. Do znalezienia najbardziej podobnych elementów pod względem semantycznym wykorzystywana jest najczęściej odległość cosinusowa (cosine similarity).

3. **Generacja Tekstu (Generation)**:
    - Pobrane *k* najbardziej podobnych elementów jest łączone z zapytaniem użytkownika.
	- Tak przygotowane zapytanie użytkownika wraz z pobranym kontekstem jest wykorzystywane przez model generacji tekstu, aby stworzyć nową, spójną odpowiedź.
	- Ta odpowiedź jest następnie przekazywana użytkownikowi jako końcowy produkt działania systemu RAG.


## Jak można ewaluować działanie systemu RAG?
Metodę RAG można ewaluować względem następujących aspektów:
- ewaluacja odzyskiwania informacji (tylko Retrieval)
    - Context Precision (dokładność kontekstu)
    - Context Recall (pełność kontekstu)
- ewaluacja generacji tekstu (tylko Generation)
    - Generation Faithfullness (wierność generacji)
    - Generation Relevance (znaczenie generacji)

![RAG - ewaluacja [5]](assets/rag_circles.png)

Chcąc ewaluować RAG całościowo, możemy patrzeć na średnią harmoniczną powyższych czterech metryk. W taki sposób działa metoda **ragas score** [4]. W tym projekcie skupimy się natomiast przede wszystkim na ewaluacji odzyskiwania informacji (Retrieval). Oczywiście zarówno aspekt Retrieval jak i Generation są niezwykle istotne, natomiast w naszym przypadku, tj. w przypadku badania wpływu różnych embeddingów, szczególnie interesującym będzie właśnie aspekt Retrieval.

Zarówno Generation Faithfullness (czyli mierzenie halucynacji) jak i Generation Relevance (czyli mierzenie jak dobrze wygenerowana odpowiedź odnosi się do zadanego pytania) zależą przede wszystkim od modelu wykorzystanego do wygenerowania ostatecznej odpowiedzi. Natomiast model tworzący embeddingi będzie miał bezpośredni wpływ na aspekt odzyskiwania informacji (Retrieval). W związku z powyższym wybierając metodę ewaluacji skupiliśmy się tylko i wyłącznie na aspekcie Retrieval.


### Wybrana metoda ewaluacji
Dokonamy badania miarą główną n-DCG@10 z miarami pomocniczymi MRR@k oraz MAP@k na zadaniu Retrieval projektu MTEB [2] na w sumie 15 benchmarkach po raz pierwszy skompilowanych na projekcie BEIR [3]. Do ewaluacji wykorzystywane jest narzędzie mteb, do podglądu rezultatów oraz oceny manualnej używany jest system RAG z wektorową bazą danych Chroma zbudowany w technologii LangChain z interfejsem w Streamlit.

## Przykładowe modele embeddingowe
- **BERT (Bidirectional Encoder Representations from Transformers)** [10]:
    - BERT jest jednym z najbardziej znanych modeli do generowania embeddingów. Jest on szczególnie efektywny w rozumieniu kontekstu, co może być kluczowe w metodzie RAG.
- **GPT-2/GPT-3 (Generative Pretrained Transformer 2/3)** [11]:
    - Te modele, znane z generowania spójnego i kontekstowego tekstu, mogą być użyteczne do badania, jak generatywne embeddingi wpływają na proces generacji w RAG.
- **RoBERTa (A Robustly Optimized BERT Pretraining Approach)** [12]:
    - RoBERTa jest wariantem BERT, który został zoptymalizowany pod kątem większej dokładności. Może to zapewnić interesujące porównanie z tradycyjnym BERT-em.
- **DistilBERT (Distilled Version of BERT)** [13]:
    - Jest to uproszczona i bardziej efektywna wersja BERT-a pod względem obliczeniowym, co może być istotne w zastosowaniach, gdzie szybkość jest kluczowa.
- **T5 (Text-To-Text Transfer Transformer)** [14]:
    - T5 jest wszechstronnym modelem, który traktuje każde zadanie NLP jako zadanie konwersji tekstu na tekst, co może być interesujące w kontekście RAG.
- **ALBERT (A Lite BERT)** [15]:
   - ALBERT to uproszczona wersja BERT-a, która została opracowana przez Google w celu radzenia sobie z problemami wynikającymi z dużych rozmiarów modelu. Używa ona technik redukcji parametrów, co może być korzystne w kontekście efektywności obliczeniowej i skalowalności systemu RAG.


Wyżej wymienione przykładowe modele w większości zostały już uwzględnione w benchmarku MTEB. W związku z tym na platformie HuggingFace postaramy się znaleźć mniej znany model, który jeszcze nie został uwzględniony w tym benchmarku, a który naszym zdaniem ma szanse uzyskać sensowne wyniki.

Co ciekawe w tabeli wyników MTEB w kategorii Retrieval wyróżnione są trzy główne podkategorie językowe - angielski, chiński oraz polski. W związku z tym ~~być może~~ ostatecznie zdecydowaliśmy skupić się na podkategorii dedykowanej dla języka polskiego.

### Ostatecznie wybrany model - **Silver Retriever Base (v1.1)**
https://huggingface.co/ipipan/silver-retriever-base-v1.1

#### Opis modelu Silver Retriever
Jest to model wytworzony przez Piotra Rybaka oraz Macieja Ogrodniczuka z Polskiej Akademii Nauk.

Model Silver Retriever bazuje na modelu HerBERT-base i został dostrojony (ang. fine-tuned) na zestawach danych PolQA i MAUPQA.

##### Zbiory danych PolQA i MAUPQA
- [PolQA](https://huggingface.co/datasets/ipipan/polqa)
- [MAUPQA](https://huggingface.co/datasets/ipipan/maupqa)


#### Dlaczego wybraliśmy akurat ten model?
Model Silver Retriever w wersji v1.1 spełnił wszystkie trzy kryteria które przede wszystkim braliśmy pod uwagę:
1. Model nie został jeszcze uwzględniony w benchmarku MTEB w kategorii Retrieval dla języka polskiego.
2. Model nie jest zbyt duży, ponieważ jako dwójka studentów nie dysponujemy ogromnymi zasobami mocy obliczeniowej.
3. Model ma szanse zrobić dobry wynik na tle pozostałych modeli.

Co ciekawe model ten w wersji v1 (czyli `silver-retriever-base-v1`) został przetestowany w benchmarku pod nazwą `herbert-base-retrieval-v2`. W związki z tym szczególnie interesujące dla nas będzie porównanie wyników tego modelu dla różnych jego wersji (tzn. v1 oraz v1.1).


#### Jakie inne modele były brane pod uwagę?
Pod uwagę brane były modele na HuggingFace spełniające następujące wymagania:
1. Obsługuje język Polski. Badanie dot. języka narodowego, będącego obecnie głównym zapotrzebowaniem komercyjnym.
2. Jest modelem kompatybilnym ze strukturą SentenceTransformers. Jest ona niezbędna do celu replikacji badania.
3. Nie znajduje się obecnie w badaniu MTEB.
4. Jest relatywnie mały. Ze względu na ograniczone zasoby, nie posiadamy sił obliczeniowych wymagannych przez większe modele.

Spośród 41 modeli najlepiej powyższe wymagania spełnił Silver Retriever.

#### Testowe zbiory danych
| Name                                                                                                                                                                  | Hub URL                                                                                                                              | Description                                                                                                                                                                                                      | Type               | Category | #Languages | Train #Samples | Dev #Samples | Test #Samples | Avg. chars / train | Avg. chars / dev | Avg. chars / test |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------- | :------- | ---------: | -------------: | -----------: | ------------: | -----------------: | ---------------: | ----------------: |
| [ArguAna-PL](http://argumentation.bplaced.net/arguana/data)                                                                                                           | [BeIR-PL/arguana-pl](https://huggingface.co/datasets/clarin-knext/arguana-pl) | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval | Retrieval | p2p | 1 | 0 | 0 | 10080 | 0 | 0 | 1052.9 |
| [DBPedia-PL](https://github.com/iai-group/DBpedia-Entity/)                                                                                                            | [BeIR-PL/dbpedia-pl](https://huggingface.co/datasets/clarin-knext/dbpedia-pl) | DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base | Retrieval | s2p | 1 | 0 | 4635989 | 4636322 | 0 | 310.2 | 310.1 |
| [FiQA-PL](https://sites.google.com/view/fiqa/)                                                                                                                        | [BeIR-PL/fiqa-pl](https://huggingface.co/datasets/clarin-knext/fiqa-pl) | Financial Opinion Mining and Question Answering | Retrieval | s2p | 1 | 0 | 0 | 58286 | 0 | 0 | 760.4 |
| [HotpotQA-PL](https://hotpotqa.github.io/) | [BeIR-PL/hotpotqa-pl](https://huggingface.co/datasets/clarin-knext/hotpotqa-pl) | HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. | Retrieval | s2p | 1 | 0 | 0 | 5240734 | 0 | 0 | 288.6 |
| [MSMARCO-PL](https://microsoft.github.io/msmarco/) | [BeIR-PL/msmarco-pl](https://huggingface.co/datasets/clarin-knext/msmarco-pl) | MS MARCO is a collection of datasets focused on deep learning in search. Note that the dev set is used for the leaderboard. | Retrieval | s2p | 1 | 0 | 8848803 | 8841866 | 0 | 336.6 | 336.8 |
| [NFCorpus-PL](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | [BeIR-PL/nfcorpus-pl](https://huggingface.co/datasets/clarin-knext/nfcorpus-pl) | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval | Retrieval | s2p | 1 | 0 | 0 | 3956 | 0 | 0 | 1462.7 |
| [NQ-PL](https://ai.google.com/research/NaturalQuestions/) | [BeIR-PL/nq-pl](https://huggingface.co/datasets/clarin-knext/nq-pl) | Natural Questions: A Benchmark for Question Answering Research | Retrieval | s2p | 1 | 0 | 0 | 2684920 | 0 | 0 | 492.7 |
| [Quora-PL](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | [BeIR-PL/quora-pl](https://huggingface.co/datasets/clarin-knext/quora-pl) | QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. | Retrieval | s2s | 1 | 0 | 0 | 532931 | 0 | 0 | 62.9 |
| [SCIDOCS-PL](https://allenai.org/data/scidocs) | [BeIR-PL/scidocs-pl](https://huggingface.co/datasets/clarin-knext/scidocs-pl) | SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. | Retrieval | s2p | 1 | 0 | 0 | 26657 | 0 | 0 | 1161.9 |
| [SciFact-PL](https://github.com/allenai/scifact) | [BeIR-PL/scifact-pl](https://huggingface.co/datasets/clarin-knext/scifact-pl) | SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts. | Retrieval | s2p | 1 | 0 | 0 | 5483 | 0 | 0 | 1422.3 |
| [TRECCOVID](https://ir.nist.gov/covidSubmit/index.html)                                                                                                               | [BeIR/trec-covid](https://huggingface.co/datasets/BeIR/trec-covid)                                                                   | TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic                                                                             | Retrieval          | s2p      |          1 |              0 |            0 |        171382 |                  0 |                0 |            1117.4 |


## Wyniki eksperymentów
Dokładne wyniki dla każdego z zadań można znaleźć w folderze `/mteb_benchmark/results/pl/silver-retriever-base-v1.1`. Wyniki dla każdego z "tasków" zawierają się w oddzielnych plikach w formacie `.json`.

Poniższa tabela jest uzupełnieniem aktualnej tabeli benchmarku mteb w kategorii retrieval dla języka polskiego. Tabela ta została uzupełniona o dodatkowy wiersz z danymi dla testowanego przez nas modeli.

Model | Average | ArguAna-PL | DBPedia-PL | FiQA-PL | HotpotQA-PL | MSMARCO-PL | NFCorpus-PL | NQ-PL | Quora-PL | SCIDOCS-PL | SciFact-PL | TRECCOVID-PL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
[mmlw-roberta-large](https://huggingface.co/sdadas/mmlw-roberta-large) | 52.71 | 63.4 | 40.27 | 40.89 | 71.04 | 36.63 | 33.94 | 47.62 | 85.51 | 19.47 | 70.23 | 70.81 |
[mmlw-e5-large](https://huggingface.co/sdadas/mmlw-e5-large) | 52.63 | 63.25 | 39.84 | 39.9 | 70.94 | 36.47 | 34.03 | 47.33 | 85.63 | 19.13 | 71.21 | 71.18 |
[mmlw-e5-base](https://huggingface.co/sdadas/mmlw-e5-base) | 50.06 | 58.4 | 37.19 | 34.53 | 66.25 | 32.54 | 33.71 | 44.6 | 84.44 | 17.35 | 68.29 | 73.33 |
[mmlw-roberta-base](https://huggingface.co/sdadas/mmlw-roberta-base) | 49.92 | 59.02 | 36.22 | 35.01 | 66.64 | 33.05 | 34.14 | 45.65 | 84.44 | 17.84 | 65.75 | 71.33 |
[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 48.98 | 53.02 | 35.82 | 33.0 | 67.41 | 33.38 | 30.24 | 52.79 | 83.65 | 13.81 | 65.66 | 70.03 |
[multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 44.01 | 42.81 | 30.23 | 25.52 | 63.52 | 29.52 | 25.98 | 44.8 | 81.22 | 12.35 | 62.11 | 66.06 |
[mmlw-e5-small](https://huggingface.co/sdadas/mmlw-e5-small) | 42.83 | 54.31 | 30.28 | 29.75 | 57.14 | 25.94 | 27.6 | 33.83 | 81.15 | 14.79 | 58.14 | 58.2 |
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 42.43 | 37.43 | 29.27 | 22.03 | 60.15 | 26.94 | 26.48 | 40.46 | 78.7 | 11.6 | 62.76 | 70.92 |
[st-polish-kartonberta-base-alpha-v1](https://huggingface.co/OrlikB/st-polish-kartonberta-base-alpha-v1) | 42.19 | 56.06 | 27.0 | 24.73 | 50.61 | 43.25 | 31.15 | 28.89 | 83.59 | 12.21 | 57.73 | 48.83 |
[st-polish-paraphrase-from-mpnet](https://huggingface.co/sdadas/st-polish-paraphrase-from-mpnet) | 34.44 | 51.87 | 24.59 | 22.27 | 32.11 | 17.91 | 24.05 | 23.54 | 81.49 | 13.23 | 52.51 | 35.23 |
[st-polish-paraphrase-from-distilroberta](https://huggingface.co/sdadas/st-polish-paraphrase-from-distilroberta) | 32.08 | 49.42 | 19.82 | 19.58 | 23.47 | 16.51 | 22.49 | 19.83 | 81.17 | 12.15 | 49.49 | 38.97 |
[paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | 29.16 | 42.62 | 20.18 | 14.68 | 29.36 | 12.45 | 18.53 | 15.64 | 79.18 | 11.18 | 41.53 | 35.38 |
[paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 26.66 | 37.83 | 18.0 | 12.49 | 22.76 | 10.39 | 17.16 | 12.56 | 77.18 | 10.26 | 40.24 | 34.38 |
[LaBSE](https://huggingface.co/sentence-transformers/LaBSE) | 23.36 | 38.52 | 16.1 | 7.63 | 19.72 | 7.22 | 17.45 | 9.65 | 74.96 | 7.48 | 39.79 | 18.45 |
[distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) | 21.18 | 36.7 | 12.36 | 8.02 | 20.83 | 4.57 | 16.28 | 5.85 | 71.95 | 6.5 | 33.03 | 16.91 |
[herbert-base-retrieval-v2](https://huggingface.co/ipipan/herbert-base-retrieval-v2) | 39.16 | 41.97 | 24.07 | 24.25 | 43.41 | 51.56 | 25.95 | 35.09 | 78.86 | 11.0 | 51.92 | 42.64 |
[silver-retriever-base-v1.1](https://huggingface.co/ipipan/silver-retriever-base-v1.1) | 37.59 | 41.72 | 23.69 | 22.07 | 38.51 | 46.32 | 24.48 | 34.65 | 77.15 | 10.87 | 49.69 | 44.31 | 


## Wnioski

## Aplikacja
W celu dokonania oglądu własnego działania wybranego modelu i weryfikacji jego możliwości został on podłączony do bazy wektorowej Chroma. Na podstawie tej bazy, utworzono standardowy interfejs, za pomocą którego użytkownik jest w stanie zapewnić źródła bazie wiedzy a następnie ją odpytać za pośrednictwem LLM. Do testowania zostały jako LLM użyte modele GPT4-turbo oraz Falcon-180b udostępniane komercyjnie odpowiednio przez OpenAI oraz IBM. Dla wygody użytkownika, dodano także możliwość połączenia się z modelami udostępnianymi na platformach WatsonX oraz HuggingFace.

W trakcie testów z pomocą aplikacji doszliśmy do następujących wniosków:
1. Model silver v1.1 jak na swoją wielkość, zachowuje się bardzo efektywnie
2. Znalezione przez model silver v1.1 informacje są w znacznej większości przypadków związane z zadanym pytaniem i prawie zawsze prawidłowo posortowane od najmniej do najbardziej istotnych.
3. Wszelkie braki w odpowiedziach aplikacji zostały ujawnione jako wina interpretujących modeli językowych. O ile komercyjne zastosowania dawały stabilne wyniki, o tyle większość z nich nie jest dostosowanych do języka polskiego.

### Technologia
Do obsługi bazy wektorowej wybrano open-source'ową bazę Chroma. Została ona przedłożona nad komercjalne rozwiązania tj. Pinecone czy DBLance głównie dzięki jej łatwej i darmowej budowie w środowisku lokalnym. Posiadając duże wsparcie społeczności, jest ona również tylko niewiele mniej wydajna od rozwiązań komercyjnych a więcej niż wystarczająco do użytkowania związanego z rozpatrywanym projektem.

Implementacją GUI został wybrany Streamlit, będących jedną z prostszych bibliotek do tego stworzonych. O ile nie posiada on znacznych możliwości regulacji, dzięki wysokiej abstrakcyjności obiektów można łatwo stworzyć powtarzalny, a mimo wszystko estetyczny interfejs. Znajduje on obecnie często zastosowanie w betatestowaniu i udostępnianiu usług świadczonych przez modele językowe.

Backend odpowiadający za przetwarzanie informacji oraz współpracę pomiędzy interfejsem, bazą wektorową (wiedzy) oraz LLM-em został opracowany w technologii LangChain. Jest to jedna z wiodących prym bibliotek udostępniających zaawansowane abstrakcje (nazywane łańcuchami) umożliwiające zarządzanie przepływem informacji w aplikacjiach opartych na modelach językowych. Zapewnia ona integrację z wielką liczbą dostawców zarówno modeli, jak i rozwiązań powiązanych. Zastosowanym łańcuchem został standardowy do takich zadań *RetrievalQA*, który kompiluje odpowiedź modelu na podstawie informacji uzyskanych z bazy wiedzy.

![RAG](assets/app.png)

## Źródła
[1] Andrew Rosenberg and Julia Hirschberg. 2007. Vmeasure: A conditional entropy-based external cluster evaluation measure. pages 410–420.

[2] Muennighoff, Niklas, Nouamane Tazi, Loïc Magne, i Nils Reimers. "MTEB: Massive Text Embedding Benchmark." ArXiv:2210.07316.

[3] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models

[4] https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/

[5] https://cobusgreyling.medium.com/rag-evaluation-9813a931b3d4

[6] https://www.linkedin.com/pulse/what-retrieval-augmented-generation-grow-right/

[7] Thulke, David, Nico Daheim, Christian Dugast and Hermann Ney. “Efficient Retrieval Augmented Generation from Unstructured Knowledge for Task-Oriented Dialog.” ArXiv abs/2102.04643 (2021): n. pag.

[8] Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

[9] "BEIR-PL: Zero Shot Information Retrieval." ArXiv:2305.19840.

[10] Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." ArXiv:1810.04805.

[11] Radford, Alec, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. "Language Models are Unsupervised Multitask Learners." OpenAI.

[12] Liu, Yinhan, Danqi Chen, et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." ArXiv:1907.11692.

[13] Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." ArXiv:1910.01108.

[14] Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." Google, Mountain View, CA. ArXiv:1910.10683.

[15] Lan, Zhenzhong, Piyush Sharma, et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." Google AI. ArXiv:1909.11942.


# dokumentacja końcowa - TODO
- [ ] opis że została przygotowana także przykładowa aplikacja wykorzystująca RAG
    - [ ] załączenie screenów z przykładowej aplikacji
- [ ] struktura kodu (podzielić na HF oraz aplikację, a następnie to ładnie opisać)
