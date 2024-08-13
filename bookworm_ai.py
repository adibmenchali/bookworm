from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Document
from haystack.components.evaluators import SASEvaluator
import requests

def setup_indexing_pipeline(model="sentence-transformers/all-MiniLM-L6-v2"):
    document_store = InMemoryDocumentStore()

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    return indexing_pipeline, document_store


def fetch_book_data(book_title):
    api_url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}&maxResults=1"
    response = requests.get(api_url)
    if response.status_code == 200:
        book_data = response.json()
        if "items" in book_data:
            return book_data["items"]
        print(book_data["items"])
    return None


# Define a global variable to store the last extracted book title
last_extracted_book_title = ""

def extract_book_title(question, ner_model):
    global last_extracted_book_title
    
    entities = ner_model(question)
    for ent in entities.ents:  # Iterate over recognized entities
        if ent.label_ == 'BOOK_TITLE':  # Check if the entity is a book title
            if ent.text.lower() == "it" and last_extracted_book_title:
                return last_extracted_book_title
            else:
                last_extracted_book_title = ent.text
                return ent.text
    return None

def create_document_objects(book_items):
    documents = []
    if book_items:
        for item in book_items:
            meta = item.get("volumeInfo", {})

            # Ensure all keys are present and set their value to "Not available" if missing
            all_keys = ['subtitle', 'authors', 'publisher', 'publishedDate',
                        'industryIdentifiers', 'readingModes', 'pageCount', 'printType', 'categories',
                        'averageRating', 'ratingsCount', 'maturityRating', 'allowAnonLogging',
                        'contentVersion', 'panelizationSummary', 'imageLinks', 'language',
                        'previewLink', 'infoLink', 'canonicalVolumeLink', 'seriesInfo']
            for key in all_keys:
                if key not in meta:
                    meta[key] = "Not available"
                elif isinstance(meta[key], list):
                    # Convert each item in the list to string
                    meta[key] = ", ".join(str(item) for item in meta[key])

            # Create content with title and description
            no_answer = 'Sorry, I am unable to answer your query right now'
            content = f"Title: {meta.get('title', no_answer)};\n"
            content += f"Writer/Author: {meta.get('authors', no_answer)};\n"
            content += f"Topic: {meta.get('description', no_answer)};\n"
            content += f"Genre , Category: {meta.get('categories', no_answer)};\n"
            content += f"Number of pages: {meta.get('pageCount', no_answer)};\n"
            content += f"Published by: {meta.get('publisher', no_answer)}"
            content += f"Publish date: {meta.get('publishedDate', no_answer)}"



            # Remove title and description from meta
            meta.pop('description', None)

            document = Document(content=content, meta=meta)
            documents.append(document)
    return documents


def setup_qa_pipeline(document_store, model='sentence-transformers/all-MiniLM-L6-v2'):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader(model='deepset/roberta-base-squad2')
    reader.warm_up()

    extractive_qa_pipeline = Pipeline()

    extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")

    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

    return extractive_qa_pipeline

def setup_eval_pipeline(authors, predictions):
    
    eval_pipeline = Pipeline()

    eval_pipeline.add_component("sas_evaluator", sas_evaluator)
    result = eval_pipeline.run(
            {
            "sas_evaluator": {"ground_truth_answers": authors,
            "predicted_answers": predictions}
        }
    )

    for evaluator in result:
        print(result[evaluator]["individual_scores"])
        
    for evaluator in result:
        print(result[evaluator]["score"])


    return extractive_qa_pipeline

def get_best_answer(qa_pipeline, user_question):
    response = qa_pipeline.run(
        data={"embedder": {"text": user_question}, "retriever": {"top_k": 3},
              "reader": {"query": user_question, "top_k": 3}}
    )

    answers = response['reader']['answers']
    for answer in answers:
        print(answer,"\n")
    
    # Filter out answers with None data
    valid_answers = [answer for answer in answers if answer.data is not None]

    # Find the answer with the highest score among valid answers
    best_answer = max(valid_answers, key=lambda x: x.score)

    # Find answers with the same title as the best answer
    same_title_answers = [answer for answer in valid_answers if answer.document.meta['title'] == best_answer.document.meta['title'] and answer.score > 0.5]
    # If there are answers with the same title, select the longest one
    if same_title_answers and ('about' in user_question.lower() or 'description' in user_question.lower()):
        best_answer = max(same_title_answers, key=lambda x: len(x.data))
        
    if best_answer.data:
        best_answer = best_answer.data[0].upper() + best_answer.data[1:] + "."

    return best_answer




