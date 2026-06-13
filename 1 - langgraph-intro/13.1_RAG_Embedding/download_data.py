import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Target directory configuration
DOWNLOAD_DIR = "financial_pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 2. Modernized headers (Using standard, non-deprecated SSL verification patterns)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# The list of institutional PDFs to harvest
PDF_FILES = [
    ("JPM_Annual_2023.pdf", "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2023.pdf"),
    ("JPM_Annual_2024.pdf", "https://www.sec.gov/Archives/edgar/data/19617/000001961725000329/annualreport-2024.pdf"),
    ("DB_Annual_2023.pdf", "https://investor-relations.db.com/files/documents/annual-reports/2023/20-F-2023.pdf"),
    ("JPM_SE_Annual_2023.pdf", "https://www.jpmorgan.com/content/dam/jpm/global/disclosures/de/english-version-of-disclosures/2023-annual-report-english.pdf"),
    ("Unilever_Annual_2024.pdf", "https://www.unilever.com/files/unilever-annual-report-on-form-20-f-2024.pdf"),
]

def download_pdf(name, url):
    """
    Downloads an individual PDF file safely, avoiding redundant downloads
    and utilizing stream-based chunk writing.
    """
    path = os.path.join(DOWNLOAD_DIR, name)

    # Idempotency check (Skip already downloaded assets)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f"⏭ Skipped (Already Exists): {name}"

    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=15, verify=True)
        # By default, requests library does not throw an exception if the server returns 404 or 500 or any other error
        # Checks HTTP status code, if response is ($200$ to $299$), it stays completely silent
        # If it is failure ($4xx$ or $5xx$), interrupts your script and throws an HTTPError exception.
        response.raise_for_status()

        with open(path, 'wb') as f:
             # Increased chunk to 16KB for faster disk I/O throughput
             # Download and process the file in precise increments of 16,384 bytes (16 KB)
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
        return f"Successfully Downloaded: {name}"

    except Exception as e:
        return f"Failed to download {name}: {e}"


# Optimization 3: Concurrency Execution Engine
def main():
    print("Starting optimized parallel ingestion engine...")

    # Using a ThreadPoolExecutor to request and download multiple files at the same exact time
    # Max_workers can be tweaked based on hardware; 4-8 is a sweet spot for casual web scraping
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all download tasks to the pool immediately
        # submit() method returns a Future object
        # Code maps these Future receipt objects as keys to their corresponding file names as values
        future_to_pdf = {executor.submit(download_pdf, name, url): name for name, url in PDF_FILES}

        # An event monitor
        # It watches your running tasks and yields a Future object the exact millisecond it finishes.
        for future in as_completed(future_to_pdf):
            # This looks inside the finished placeholder task and extracts whatever your download_pdf function returned
            result_message = future.result()
            print(result_message)

if __name__ == "__main__":
    main()