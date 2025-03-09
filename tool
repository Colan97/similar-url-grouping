import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
import io
import base64
import time
from tqdm import tqdm
import concurrent.futures
import functools

# Set page config
st.set_page_config(
    page_title="URL Grouping Tool",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data silently on first run
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Cache the stemmer for performance
@st.cache_resource
def get_stemmer():
    return PorterStemmer()

# Optimized URL normalization with caching
@functools.lru_cache(maxsize=10000)
def normalize_url(url):
    """
    Normalize a URL by:
    - converting to lowercase
    - removing protocol (http://, https://)
    - removing 'www.'
    - removing trailing slash
    - removing extraneous spaces
    """
    url = url.strip().lower()
    url = re.sub(r'^https?:\/\/', '', url)  # remove http:// or https:// at start
    url = re.sub(r'^www\.', '', url)       # remove leading 'www.'
    url = url.rstrip('/')                  # remove trailing slash
    url = url.strip()
    return url

# Cached function to normalize and tokenize URLs
@functools.lru_cache(maxsize=10000)
def normalize_and_tokenize(url):
    """Normalize and tokenize a URL, returning a frozenset of stemmed tokens"""
    norm_url = normalize_url(url)
    tokens = re.split(r'[^a-z0-9]+', norm_url)
    stemmer = get_stemmer()
    return frozenset(stemmer.stem(token) for token in tokens if token)

def process_urls_optimized(urls, similarity_threshold=0.85, use_parallel=False, max_workers=None):
    """
    Optimized version of process_urls with early filtering and optional parallel processing.
    """
    # Filter empty URLs and remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        url = url.strip()
        if url and url not in seen:
            unique_urls.append(url)
            seen.add(url)
    
    urls = unique_urls
    total_urls = len(urls)
    
    if total_urls == 0:
        return {}
    
    # Pre-compute token sets
    token_sets = []
    
    with st.spinner("Pre-processing URLs..."):
        progress_bar = st.progress(0)
        for i, url in enumerate(urls):
            token_sets.append(normalize_and_tokenize(url))
            progress_bar.progress((i + 1) / total_urls)
    
    # Function to process a chunk of URLs
    def process_chunk(start_idx, end_idx):
        local_groups = defaultdict(list)
        local_group_counter = 0
        url_to_group = {}
        
        for i in range(start_idx, min(end_idx, total_urls)):
            if i in url_to_group:
                continue
                
            local_group_counter += 1
            current_group = [urls[i]]
            url_to_group[i] = local_group_counter
            current_tokens = token_sets[i]
            
            for j in range(i + 1, total_urls):
                if j in url_to_group:
                    continue
                    
                other_tokens = token_sets[j]
                
                # Quick size check before computing similarity
                if abs(len(current_tokens) - len(other_tokens)) / max(len(current_tokens), len(other_tokens), 1) > (1 - similarity_threshold):
                    continue
                
                # Compute similarity
                intersection = len(current_tokens.intersection(other_tokens))
                union = len(current_tokens.union(other_tokens))
                similarity = intersection / union if union else 0.0
                
                if similarity >= similarity_threshold:
                    current_group.append(urls[j])
                    url_to_group[j] = local_group_counter
            
            local_groups[local_group_counter] = current_group
        
        return local_groups
    
    # Main processing logic
    groups = defaultdict(list)
    
    with st.spinner("Grouping similar URLs..."):
        if use_parallel and total_urls > 500 and max_workers:
            # Parallel processing for large datasets
            chunk_size = total_urls // max_workers
            if chunk_size < 10:  # Ensure reasonable chunk sizes
                chunk_size = 10
                max_workers = max(1, total_urls // chunk_size)
                
            chunks = [(i, min(i + chunk_size, total_urls)) for i in range(0, total_urls, chunk_size)]
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor for Streamlit compatibility
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(process_chunk, start, end): (start, end) for start, end in chunks}
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_chunk):
                    start, end = future_to_chunk[future]
                    chunk_groups = future.result()
                    
                    # Merge chunk groups into the main groups
                    # This is a simplified merge - in a real scenario, you might need to merge overlapping groups
                    offset = len(groups)
                    for group_id, urls_in_group in chunk_groups.items():
                        groups[offset + group_id] = urls_in_group
                    
                    completed += 1
                    progress_text.text(f"Processed {completed}/{len(chunks)} chunks")
                    progress_bar.progress(completed / len(chunks))
        else:
            # Sequential processing
            result = process_chunk(0, total_urls)
            groups = result
    
    # Convert defaultdict to regular dict for return
    return dict(groups)

def get_download_link(df, filename="url_groups.csv", text="Download CSV"):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("URL Grouping Tool")
    st.markdown("""
    This tool helps you identify and group similar URLs based on their content and structure. 
    Useful for SEO analysis, duplicate content detection, and website organization.
    """)
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.7, 
        max_value=0.95, 
        value=0.85, 
        step=0.01,
        help="Higher values require URLs to be more similar to be grouped together"
    )
    
    use_parallel = st.sidebar.checkbox(
        "Use Parallel Processing", 
        value=True,
        help="Enable for faster processing with large URL lists (>500 URLs)"
    )
    
    if use_parallel:
        max_workers = st.sidebar.slider(
            "Max Workers", 
            min_value=2, 
            max_value=8, 
            value=4,
            help="Number of parallel processing threads"
        )
    else:
        max_workers = None
        
    # Input selection
    st.header("Input URLs")
    input_option = st.radio(
        "Choose input method:",
        ["Paste URLs", "Upload File"]
    )
    
    urls = []
    
    if input_option == "Paste URLs":
        url_text = st.text_area(
            "Paste your URLs below (one per line or separated by spaces):",
            height=200
        )
        if url_text:
            # Split by whitespace and newlines
            urls = re.split(r'[\s]+', url_text)
            urls = [u for u in urls if u.strip()]
    
    else:  # Upload File
        uploaded_file = st.file_uploader("Upload a file with URLs (one URL per line)", type=["txt", "csv"])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode()
            urls = [line.strip() for line in content.splitlines() if line.strip()]
    
    # Process button
    if urls:
        st.write(f"Found {len(urls)} URLs")
        
        if st.button("Group Similar URLs"):
            start_time = time.time()
            
            # Process URLs
            with st.spinner("Processing URLs..."):
                groups = process_urls_optimized(
                    urls, 
                    similarity_threshold,
                    use_parallel=use_parallel,
                    max_workers=max_workers
                )
            
            # Calculate statistics
            processing_time = time.time() - start_time
            num_groups = len(groups)
            
            # Create DataFrame for display
            rows = []
            for group_id, group_urls in groups.items():
                for url in group_urls:
                    rows.append({'Group': group_id, 'URL': url})
            
            if rows:
                df = pd.DataFrame(rows)
                df = df.sort_values(['Group', 'URL'])
                
                # Display stats
                st.success(f"Processing complete! Found {num_groups} groups in {processing_time:.2f} seconds.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total URLs", len(urls))
                col2.metric("Groups Found", num_groups)
                col3.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Display results
                st.header("Results")
                
                # Group statistics
                group_sizes = df.groupby('Group').size().reset_index(name='Count')
                group_sizes = group_sizes.sort_values('Count', ascending=False)
                
                st.subheader("Group Sizes")
                st.bar_chart(group_sizes.set_index('Group'))
                
                # Display groups
                st.subheader("URL Groups")
                
                # Use expanders for each group to save space
                for group_id in sorted(groups.keys()):
                    group_urls = groups[group_id]
                    with st.expander(f"Group {group_id} ({len(group_urls)} URLs)"):
                        for url in group_urls:
                            st.text(url)
                
                # Download link
                st.markdown(get_download_link(df), unsafe_allow_html=True)
            else:
                st.warning("No URL groups were found. Try adjusting the similarity threshold.")
    else:
        st.info("Please enter or upload URLs to begin.")
    
    # Add footer with GitHub link
    st.markdown("---")
    st.markdown(
        "Developed by SEO Experts | [View on GitHub](https://github.com/your-username/url-grouping-tool)"
    )

if __name__ == "__main__":
    main()
