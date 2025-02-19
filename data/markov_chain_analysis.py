import networkx as nx
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from collections import Counter
import multiprocessing as mp
import pandas as pd
import re
from datasets import load_dataset

def process_chunk(chunk):
    """Process a chunk of text to extract word pairs"""
    words = chunk.split()
    # Create word pairs within the chunk
    pairs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    # Count words in this chunk
    word_counts = Counter(words)
    return pairs, word_counts

def merge_counters(counter_list):
    """Merge multiple Counter objects"""
    final_counter = Counter()
    for c in counter_list:
        final_counter.update(c)
    return final_counter

def chunk_text(text, n_chunks):
    """Split text into roughly equal chunks"""
    words = text.split()
    chunk_size = len(words) // n_chunks
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Moving the edge weighting function outside
def process_edge(args):
    """Process a single edge with its weight"""
    pair, word_freq, min_connections, pair_counts = args
    w1, w2 = pair
    if word_freq[w1] >= min_connections and word_freq[w2] >= min_connections:
        weight = pair_counts.get((w1, w2), 0) + pair_counts.get((w2, w1), 0)
        if weight > 0:
            return (w1, w2, {'weight': weight})
    return None

def create_arabic_network_parallel(text, min_connections=2, n_processes=None):
    """Create network visualization using parallel processing"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split text into chunks
    chunks = chunk_text(text, n_processes)
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Process chunks in parallel
        results = pool.map(process_chunk, chunks)
        
        # Combine results
        all_pairs = []
        all_counters = []
        for pairs, counter in results:
            all_pairs.extend(pairs)
            all_counters.append(counter)
        
        # Merge word counts
        word_freq = merge_counters(all_counters)
        
        # Count pair frequencies
        pair_counts = Counter(all_pairs)
        
        # Create network
        G = nx.Graph()
        
        # Prepare arguments for parallel processing
        unique_pairs = set(tuple(sorted(pair)) for pair in all_pairs)
        edge_args = [(pair, word_freq, min_connections, pair_counts) 
                    for pair in unique_pairs]
        
        # Process edges in parallel
        edges = pool.map(process_edge, edge_args)
        edges = [e for e in edges if e is not None]
        
        # Add edges to graph
        G.add_edges_from(edges)
    
    # Calculate centrality
    centrality = nx.degree_centrality(G)
    
    # Visualization
    plt.figure(figsize=(20, 20), facecolor='black')
    plt.style.use('dark_background')
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_color=[centrality[node] for node in G.nodes()],
                                 node_size=10,
                                 cmap=plt.cm.YlOrRd)
    
    edges = nx.draw_networkx_edges(G, pos,
                                 edge_color='gray',
                                 alpha=0.2)
    
    # Add labels with proper Arabic text rendering
    labels = {node: get_display(arabic_reshaper.reshape(node)) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='white')
    
    # Add colorbar
    plt.colorbar(nodes, label='Centrality Score')
    
    plt.axis('off')
    return plt, G

def analyze_network(G):
    """Analyze network properties"""
    # Calculate various centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G)
    
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame({
        'Degree Centrality': degree_cent,
        'Betweenness Centrality': betweenness_cent,
        'Eigenvector Centrality': eigenvector_cent
    })
    
    return metrics_df.sort_values('Degree Centrality', ascending=False)

def filter_network_by_centrality(G, threshold=0.1):
    """
    Filter network to keep only nodes with degree centrality above threshold.
    
    Args:
        G (networkx.Graph): Input graph
        threshold (float): Minimum degree centrality value (0 to 1)
    
    Returns:
        networkx.Graph: Filtered graph
    """
    # Calculate degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # Get nodes to remove (those below threshold)
    nodes_to_remove = [node for node, cent in degree_cent.items() 
                      if cent < threshold]
    
    # Create a copy of the graph
    filtered_G = G.copy()
    
    # Remove nodes
    filtered_G.remove_nodes_from(nodes_to_remove)
    
    return filtered_G

def clean_text(text):
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    words = text.lower().split()
    return " ".join([word for word in words])

if __name__ == "__main__":
    
    MIN_CONNECTIONS = 5
    CENTRALITY_THRESHOLD = 0.05
    TOTAL_SAMPLES = 10_000
    
    train_ds = load_dataset("atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset", split='train').shuffle(seed=1998).select(range(TOTAL_SAMPLES))
    cleaned_text = " ".join(clean_text(text) for text in train_ds["text"])

    plt, G = create_arabic_network_parallel(
            cleaned_text,
            min_connections=MIN_CONNECTIONS,
            n_processes=mp.cpu_count()
    )
    
    # Filter the network and create visualization as before...
    filtered_G = filter_network_by_centrality(G, CENTRALITY_THRESHOLD)
    
    plt.figure(figsize=(20, 20), facecolor='black')
    plt.style.use('dark_background')
    
    centrality = nx.degree_centrality(filtered_G)
    pos = nx.spring_layout(filtered_G, k=1, iterations=50)
    
    nodes = nx.draw_networkx_nodes(filtered_G, pos,
                                 node_color=[centrality[node] for node in filtered_G.nodes()],
                                 node_size=10,
                                 cmap=plt.cm.YlOrRd)
    
    edges = nx.draw_networkx_edges(filtered_G, pos,
                                 edge_color='gray',
                                 alpha=0.2)
    
    labels = {node: get_display(arabic_reshaper.reshape(node)) for node in filtered_G.nodes()}
    nx.draw_networkx_labels(filtered_G, pos, labels, font_size=12, font_color='white')
    
    plt.colorbar(nodes, label='Centrality Score')
    plt.axis('off')
    plt.savefig(f'net_min_conn_{MIN_CONNECTIONS}_thrld_{CENTRALITY_THRESHOLD}_nsamples_{TOTAL_SAMPLES}.png', 
                facecolor='black', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()