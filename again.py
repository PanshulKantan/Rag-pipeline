from pypdf import PdfReader
reader=PdfReader("2026364565.pdf") # we cant iterate through it as it is complete extraction of the pdf, its not distributed which we can iterate through
page=reader.pages   #now it has been divided into pages
all_text=""
for pages in page:  # this is the loop through which we iterate 
    text=pages.extract_text()
  #  print(text)
    all_text+=text + "\n"


# print(all_text) #it doesnt show entire text together because terminal truncates it but its stored in all_text fs
# with open("output.txt", "w", encoding="utf-8") as f:
#     f.write(all_text)
# print("Text saved to output.txt")


from langchain_text_splitters import RecursiveCharacterTextSplitter
chunk_parameter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=chunk_parameter.split_text(all_text)
# print(len(chunks))
# print(chunks[7])


from sentence_transformers import SentenceTransformer
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=embedding_model.encode(chunks)


import numpy as np
embeddings = np.array(embeddings).astype("float32")   # NOT IMPORTANT BUT JUST TO ENSURE FLOAT IS THE FORMAT OF THE MATRIX


# print(len(embeddings))
# print(embeddings[0])
print(embeddings.shape)   #IMPORTANT, it will give 30x384


import faiss
database=faiss.IndexFlatL2(384) #makes the storage where the vectors will stay
database.add(embeddings)
print(database.ntotal) #tells how many vectors are being stored in the database


query=str(input("Your question: "))

embedded_q=embedding_model.encode([query])  # it has to be a list because the embedding of the data was done in chunks which was a list
#embedded_q=embedding_model.encode(query) can be written like this if query is a list of more than 1 strings as question
embedded_q=np.array(embedded_q).astype("float32")   #just to ensure its of type float 32


distances, indices=database.search(embedded_q, 3)   #the indices and distances here is returned in 2d array format for handling more than one question
print(indices)
# print(distances)
# for i in indices[0]:
#     print(chunks[i])
#     print("-"*80)       #yeah just for showing the relevant chunks and breaking them with the -------------- line


retrieved_chunk=[]
for i in indices[0]:
    retrieved_chunk.append(chunks[i])
prompt_context="\n\n".join(retrieved_chunk)


from google import genai
gemini_client=genai.Client(api_key="#####################")



prompt = f"""
You are an AI assistant.

Answer ONLY from the provided context, Start with greeting me, my name is Panshul Kantan, i am the author of this research paper

If the answer is not available,
say "I don't know."

Context:

{prompt_context}

Question:

{query}

Answer:
"""

answer=gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)
print(answer.text)
# .venv\Scripts\activate.bat