
"""
uv run embedding.py
"""
import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_setup import get_gemini_embedding_model


load_dotenv(override=True)


def assign_roles_to_files(files):
    """Assign roles based on order: 1st analyst, 2nd scientist, 3rd financial."""
    role_map = {}
    for i, file in enumerate(files):
        if i == 0:
            role_map[file] = "analyst"
        elif i == 1:
            role_map[file] = "scientist"
        else:
            role_map[file] = "financial"
    return role_map


def load_data_from_folder(folder_path):
    """Load all PDFs from a folder and attach file_name + role metadata with logs."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pdf")])
    role_mapping = assign_roles_to_files(files)

    documents = []
    for file in files:
        pdf_path = os.path.join(folder_path, file)
        assigned_role = role_mapping[file]
        print(f"Processing Text Layer: {file} | Core Meta-Tag: {assigned_role}")

        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Update metadata
            # # Injection Point: Bind processing properties directly to structural dictionaries
            for d in docs:
                d.metadata["file_name"] = file
                d.metadata["role"] = assigned_role
            documents.extend(docs)
            print(f" Successfully indexed {len(docs)} pages.")
        except Exception as e:
            print(f" Failed parsing document structure for {file}: {e}")

    return documents


def chunk_documents(documents, chunk_size_tokens=500, chunk_overlap_tokens=50):
    """
    Slices raw documents using optimized token approximations.
    500 tokens is highly performant for balancing context and semantic retrieval precision.
    """
    # 1 Token equates roughly to 4 english characters
    chunk_size_chars = chunk_size_tokens * 4
    chunk_overlap_chars = chunk_overlap_tokens * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def create_and_save_faiss_batched(documents, embedding_model, save_path, batch_size=10, sleep_time=3):
    """
    Streams content chunks down to vector indexes safely, featuring an automated loop retry mechanism
    to handle rate limit exceptions without data loss.
    """
    vectorstore = None
    total_chunks = len(documents)

    print(f"\n⚡ Streaming {total_chunks} chunks to Vector Database (Batch Size: {batch_size})...")

    i = 0
    while i < total_chunks:
        batch = documents[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1

        try:
            # Build database index slice for this temporary target block
            db_slice = FAISS.from_documents(batch, embedding_model)

            if vectorstore is None:
                vectorstore = db_slice
            else:
                vectorstore.merge_from(db_slice)

            print(f" Completed processing batch {current_batch_num}")

            i += batch_size
            time.sleep(sleep_time) # Smooth API cooldown spacer

        except Exception as e:
            # FIX: Do not drop data. Warn the operator, back off, and rerun the same index segment again.
            print(f"   ⚠️ Rate limit or network hiccup on batch {current_batch_num}: {e}")
            print(f"      Retrying exact segment in {sleep_time * 2} seconds to protect pipeline integrity...")
            time.sleep(sleep_time * 2)

    if vectorstore:
        vectorstore.save_local(save_path)
        print(f"\n💾 Production FAISS database successfully deployed locally to: {save_path}")
    else:
        print("\n❌ Error: No vectors were constructed.")

if __name__ == '__main__':
    # 1. Establish the exact absolute path where this embedding.py script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    # 2. Define our targets relative to workspace realities
    RAG_PDF_FOLDER = "financial_pdfs"
    VECTOR_DB_DIR = "13.1_RAG_Embedding"
    VECTOR_DB_SUBDIR = os.path.join(VECTOR_DB_DIR, "faiss_index_financial")

    # 3. Dynamic Path Resolver Strategy Engine
    # Option A: Check if financial_pdfs is in the same folder as the script
    if os.path.exists(os.path.join(SCRIPT_DIR, RAG_PDF_FOLDER)):
        TARGET_INPUT_FOLDER = os.path.join(SCRIPT_DIR, RAG_PDF_FOLDER)
        VECTOR_OUTPUT_DB = os.path.join(SCRIPT_DIR, VECTOR_DB_SUBDIR)

    # Option B: Check if financial_pdfs lives in the Parent directory (e.g., you are nested in a subdir)
    elif os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), RAG_PDF_FOLDER)):
        PARENT_DIR = os.path.dirname(SCRIPT_DIR)
        TARGET_INPUT_FOLDER = os.path.join(PARENT_DIR, RAG_PDF_FOLDER)
        # Keeps your vector database close to the code execution directory
        VECTOR_OUTPUT_DB = os.path.join(SCRIPT_DIR, "faiss_index_financial")

    # Option C: Check if it's nested down in a subdirectory below you
    elif os.path.exists(os.path.join(SCRIPT_DIR, VECTOR_DB_DIR, RAG_PDF_FOLDER)):
        TARGET_INPUT_FOLDER = os.path.join(SCRIPT_DIR, VECTOR_DB_DIR, RAG_PDF_FOLDER)
        VECTOR_OUTPUT_DB = os.path.join(SCRIPT_DIR, VECTOR_DB_SUBDIR)

    # Option D: Safe Failure Fallback — create it right here in the workspace to prevent a crash
    else:
        print(f"⚠️ Could not find an existing '{RAG_PDF_FOLDER}' folder layout.")
        TARGET_INPUT_FOLDER = os.path.join(SCRIPT_DIR, RAG_PDF_FOLDER)
        VECTOR_OUTPUT_DB = os.path.join(SCRIPT_DIR, VECTOR_DB_SUBDIR)
        os.makedirs(TARGET_INPUT_FOLDER, exist_ok=True)
        print(f"📁 Created a new data landing pad at: {TARGET_INPUT_FOLDER}")

    print(f"Pipeline Path Resolved -> PDF Source Folder: {TARGET_INPUT_FOLDER}")
    print(f"Pipeline Path Resolved -> Local Vector Store: {VECTOR_OUTPUT_DB}")

    raw_docs = load_data_from_folder(TARGET_INPUT_FOLDER)

    if not raw_docs:
        print(f"Aborting pipeline run: Zero valid text chunks found inside '{TARGET_INPUT_FOLDER}'. Please drop your source PDFs there.")
    else:
        print(f"\nTotal Pages Successfully Harvested: {len(raw_docs)}\n")

        # 4. Processing/Slicing Phase (Optimized ~500 Tokens Window Size)
        processed_chunks = chunk_documents(raw_docs, chunk_size_tokens=500, chunk_overlap_tokens=50)
        print(f"Sliced content into {len(processed_chunks)} high-precision text nodes.")

        # 5. Production Local FAISS Index Streaming Deployment
        embedding_engine = get_gemini_embedding_model()
        create_and_save_faiss_batched(
            documents=processed_chunks,
            embedding_model=embedding_engine,
            save_path=VECTOR_OUTPUT_DB,
            batch_size=10,
            sleep_time=2
        )