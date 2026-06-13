"""
uv run inspect_db.py
"""
import os
from langchain_community.vectorstores import FAISS
from embedding_setup import get_gemini_embedding_model

def inspect_local_faiss_store(faiss_index_path):
    # 1. Load the model to use as a semantic reference validator
    embedding_model = get_gemini_embedding_model()

    print(f"Opening local FAISS database index at: {faiss_index_path}\n")

    # 2. Load the binary files back into volatile system RAM memory
    # allow_dangerous_deserialization=True is required to read local pickle (.pkl) metadata maps
    db = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    # ==========================================
    # LAYER 1: Inspecting the Python Docstore Map
    # ==========================================
    print("==================================================")
    # Extract the internal private dictionary map from the LangChain wrapper
    id_to_doc_map = db.docstore._dict
    print(f"📊 Total document chunks tracked in Python Docstore: {len(id_to_doc_map)}")

    # Let's peek at the entries inside our dictionary map
    for internal_id, doc_obj in list(id_to_doc_map.items())[:2]:
        print(f"\n🔹 LangChain Internal Tracking Key ID: '{internal_id}'")
        print(f"   📄 Source Document: {doc_obj.metadata.get('file_name')}")
        print(f"   👤 Assigned Agent Filtering Role: {doc_obj.metadata.get('role')}")
        print(f"   📝 Content Snippet: \"{doc_obj.page_content[:90]}...\"")

    # ==========================================
    # LAYER 2: Extracting the Hidden Raw Vectors
    # ==========================================
    print("\n==================================================")
    print("🔢 EXTRACTING THE HIDDEN COORDINATE MATRIX VECTORS")
    print("==================================================")

    # Access the underlying native C++ FAISS index tracking layout
    native_faiss_index = db.index

    # LangChain maps external string tracking IDs to clean sequential integer IDs
    # Let's trace the index keys that correspond to our document mapping values
    index_to_docstore_id = db.index_to_docstore_id

    # Iterate through our entries to extract the corresponding vector matrices
    for matrix_id in range(min(2, native_faiss_index.ntotal)):
        # Reconstruct the vector coordinates back out of the index graph map matrix
        # reconstructed_vector is a raw numpy array of floats
        reconstructed_vector = native_faiss_index.reconstruct(matrix_id)

        # Pull out the corresponding tracking key from our index array map
        docstore_id = index_to_docstore_id[matrix_id]
        matching_text_chunk = id_to_doc_map[docstore_id].page_content[:50].replace('\n', ' ')

        print(f"\n📍 Coordinate Matrix Vector ID Slot: {matrix_id} (Maps to tracking key: '{docstore_id}')")
        print(f"   🔤 Start of Text Chunk content: \"{matching_text_chunk}...\"")
        print(f"   📏 Vector Space Footprint Size: {len(reconstructed_vector)} Dimensions")

        # Format a string array segment to print out the first few raw float values
        float_sample = ", ".join([f"{num:.6f}" for num in reconstructed_vector[:6]])
        print(f"   🔢 Raw Floating-Point Vector Coordinates Array: [{float_sample}, ... (762 more numbers)]")


if __name__ == '__main__':
    # Determine the script's working context location dynamically
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    # Target path pointing straight to your saved local FAISS index folder layout
    TARGET_FAISS_DB = os.path.join(BASE_DIR, "RAG_embedding", "faiss_index_financial")

    if not os.path.exists(TARGET_FAISS_DB):
        # Fallback look if you are executing directly from inside the 13.1 subdir
        TARGET_FAISS_DB = os.path.join(BASE_DIR, "faiss_index_financial")

    if os.path.exists(TARGET_FAISS_DB):
        inspect_local_faiss_store(TARGET_FAISS_DB)
    else:
        print(f"❌ Error: Could not find your faiss_index_financial folder at: {TARGET_FAISS_DB}")