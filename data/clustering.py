import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.colors as mcolors
import json
import logging
import os
import textwrap
from collections import Counter, defaultdict
import faiss
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP
from sklearn.manifold import TSNE
from openai import OpenAI
import time
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO)


DEFAULT_INSTRUCTION = (
    instruction
) = """
Your task is to describe the common theme or topic in the above text samples in a single, concise sentence. Keep your sentence under 5-7 words and do not say anything else but the topic and do not add any explanation or leading sentence!!
You SHOULD use english to describe the topics.\
Example of topics include (but not limited to): Sociology, Politics, News, Moroccan cooking recipes, Hair care receipes, Islamic Fatwas on fasting, Islamic Fatwas on mariage, etc.
"""

DEFAULT_TEMPLATE = "<s>[INST]{examples}\n\n{instruction}[/INST]"


class ClusterClassifier:
    def __init__(
        self,
        embed_model_name="all-MiniLM-L6-v2",
        embed_device="cpu",
        embed_batch_size=64,
        embed_max_seq_length=512,
        embed_agg_strategy=None,
        n_components=2,
        method="tsne",
        dbscan_eps=0.08,
        dbscan_min_samples=50,
        dbscan_n_jobs=16,
        summary_create=True,
        summary_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        topic_mode="multiple_topics",
        summary_n_examples=10,
        summary_chunk_size=420,
        summary_model_token=True,
        summary_template=None,
        summary_instruction=None,
        use_gemini=False,
        cluster_projected=True,
        gemini_token="",
        max_retry=6,
        matryoshka_dim=64,
        **kwargs

    ):
        
        self.kwargs = kwargs
        self.use_gemini=use_gemini
        self.max_retry=max_retry
        self.gemini_token=gemini_token
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy
        self.embed_dim = matryoshka_dim
        self.cluster_projected = cluster_projected

        self.method = method.lower()
        self.n_components = n_components
        
        if self.method == 'tsne':
            self.mapper = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000),
                random_state=42
            )
        elif self.method == 'umap':
            self.mapper = UMAP(
                n_components=n_components,
                metric=kwargs.get('metric', 'euclidean'),
                random_state=42
            )
        else:
            raise ValueError("Method must be either 'tsne' or 'umap'")

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_n_jobs = dbscan_n_jobs

        self.summary_create = summary_create
        self.summary_model = summary_model
        self.topic_mode = topic_mode
        self.summary_n_examples = summary_n_examples
        self.summary_chunk_size = summary_chunk_size
        self.summary_model_token = summary_model_token

        if summary_template is None:
            self.summary_template = DEFAULT_TEMPLATE
        else:
            self.summary_template = summary_template

        if summary_instruction is None:
            self.summary_instruction = DEFAULT_INSTRUCTION
        else:
            self.summary_instruction = summary_instruction

        self.embeddings = None
        self.faiss_index = None
        self.cluster_labels = None
        self.texts = None
        self.projections = None
        self.umap_mapper = None
        self.id2label = None
        self.label2docs = None

        self.embed_model = SentenceTransformer(
            self.embed_model_name, device=self.embed_device, truncate_dim=self.embed_dim,
        )
        self.embed_model.max_seq_length = self.embed_max_seq_length

    def fit(self, texts, embeddings=None):
        self.texts = texts

        if embeddings is None:
            logging.info("embedding texts...")
            self.embeddings = self.embed(texts)
        else:
            logging.info("using precomputed embeddings...")
            self.embeddings = embeddings

        logging.info("building faiss index...")
        self.faiss_index = self.build_faiss_index(self.embeddings)
        logging.info(f"projecting with {self.method}...")
        self.projections, self.umap_mapper = self.project(self.embeddings)
        
        if self.cluster_projected:
            logging.info("dbscan clustering on projected embeddings...")
            self.cluster_labels = self.cluster(self.projections)
        else:
            logging.info("dbscan clustering on embeddings...")
            self.cluster_labels = self.cluster(self.embeddings)
        
        logging.info(f"Found {len(set(self.cluster_labels))} clusters")

        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

        if self.summary_create:
            logging.info("summarizing cluster centers...")
            self.cluster_summaries = self.summarize(self.texts, self.cluster_labels)
        else:
            self.cluster_summaries = None

        return self.embeddings, self.cluster_labels, self.cluster_summaries

    def infer(self, texts, top_k=1):
        embeddings = self.embed(texts)

        dist, neighbours = self.faiss_index.search(embeddings, top_k)
        inferred_labels = []
        for i in tqdm(range(embeddings.shape[0])):
            labels = [self.cluster_labels[doc] for doc in neighbours[i]]
            inferred_labels.append(Counter(labels).most_common(1)[0][0])

        return inferred_labels, embeddings

    def embed(self, texts):
        embeddings = self.embed_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings
    
    def project(self, embeddings):
        """Project embeddings to lower dimensional space"""
        if self.method.upper() == 'TSNE':
            embedding = self.mapper.fit_transform(embeddings)
            return embedding, self.mapper
        elif self.method.upper() == 'UMAP':
            self.mapper.fit(embeddings)
            return self.mapper.embedding_, self.mapper
        else:
            raise NotImplementedError(f"Method {self.method} not implemented. Choose 'tsne' or 'umap'.")

    def cluster(self, embeddings):
        print(
            f"Using DBSCAN (eps, nim_samples)=({self.dbscan_eps,}, {self.dbscan_min_samples})"
        )
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=self.dbscan_n_jobs,
        ).fit(embeddings)

        return clustering.labels_

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def summarize(self, texts, labels):
        unique_labels = len(set(labels)) - 1  # exclude the "-1" label
        cluster_summaries = {-1: "None"}
        if self.use_gemini:
          client=OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key=self.gemini_token
          )
        else:
          client = InferenceClient(self.summary_model, token=self.summary_model_token)

        for i,label in tqdm(enumerate(range(unique_labels)), desc="Summarizing..."):
            ids = np.random.choice(self.label2docs[label], self.summary_n_examples)
            examples = "\n\n".join(
                [
                    f"Example {i+1}:\n{texts[_id][:self.summary_chunk_size]}"
                    for i, _id in enumerate(ids)
                ]
            )
            if self.use_gemini:
              count=0
              while True:
                try:
                  request=f"{examples}\n\n{instruction}"
                  response=client.chat.completions.create(
                    messages=[
                      {"role":"user",
                      "content":request}
                    ],
                    model="gemini-2.0-flash-exp"
                  ).choices[0].message.content
                  break
                except Exception as e:
                  print(f"[INFO] error {e}")
                  delay=60
                  count+=1
                  if count==self.max_retry:
                    raise e
                  else:
                    time.sleep(delay)
                    continue
            else: 
              request = self.summary_template.format(
                  examples=examples, instruction=self.summary_instruction
              )
              response = client.text_generation(request)
              print(f'topic response: {response}')
            cluster_summaries[label] = response
        return cluster_summaries

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        faiss.write_index(self.faiss_index, f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "wb") as f:
            np.save(f, self.projections)

        with open(f"{folder}/cluster_labels.npy", "wb") as f:
            np.save(f, self.cluster_labels)

        with open(f"{folder}/texts.json", "w") as f:
            json.dump(self.texts, f)

        with open(f"{folder}/mistral_prompt.txt", "w") as f:
            f.write(DEFAULT_INSTRUCTION)

        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)

    def load(self, folder):
        print("[INFO] Loading pre-computed clusters...")
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exsit.")

        with open(f"{folder}/embeddings.npy", "rb") as f:
            self.embeddings = np.load(f)

        self.faiss_index = faiss.read_index(f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "rb") as f:
            self.projections = np.load(f)

        with open(f"{folder}/cluster_labels.npy", "rb") as f:
            self.cluster_labels = np.load(f)

        with open(f"{folder}/texts.json", "r") as f:
            self.texts = json.load(f)

        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # those objects can be inferred and don't need to be saved/loaded
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def show_old(self, interactive=False):
        df = pd.DataFrame(
            data={
                "X": self.projections[:, 0],
                "Y": self.projections[:, 1],
                "labels": self.cluster_labels,
                "content_display": [
                    textwrap.fill(txt[:1024], 64) for txt in self.texts
                ],
            }
        )

        if interactive:
            self._show_plotly(df)
        else:
            self._show_mpl(df)
            
    def show(self, interactive=False, figure_style="paper", dim_3d=False, show_summaries=True, include_noise_points=True):
        """
        Enhanced visualization method for clustering results
        Args:
            interactive (bool): Whether to use plotly (True) or matplotlib (False)
            figure_style (str): Either "paper" or "default" for different styling options
            dim_3d (bool): Whether to show 3D visualization (only works with interactive=True)
        """
        # Check if 3D is possible
        if dim_3d and self.n_components < 3:
            print("Warning: 3D visualization requested but n_components < 3. Falling back to 2D.")
            dim_3d = False

        # Create DataFrame with appropriate dimensions
        data_dict = {
            "X": self.projections[:, 0],
            "Y": self.projections[:, 1],
            "labels": self.cluster_labels,
            "content_display": [
                textwrap.fill(txt[:1024], 64) for txt in self.texts
            ],
        }
        
        # Add Z coordinate if using 3D
        if dim_3d and self.projections.shape[1] >= 3:
            data_dict["Z"] = self.projections[:, 2]
        
        df = pd.DataFrame(data=data_dict)

        if interactive:
            if dim_3d:
                fig = self._show_plotly_3d(df, style=figure_style)
            else:
                fig = self._show_plotly_enhanced(df, style=figure_style, show_summaries=show_summaries, include_noise_points=include_noise_points)
        else:
            if dim_3d:
                print("3D visualization is only available with interactive=True")
            fig = self._show_mpl_enhanced(df, style=figure_style)
            
        return fig

    def _show_mpl_enhanced(self, df, style="paper"):
        """Enhanced matplotlib visualization with soft, gradient-like cluster boundaries"""
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Custom color palette with pastel colors
        colors = {
            -1: "#E6E6E6",  # noise color
            0: "#FFB6C1",   # light pink
            1: "#98FB98",   # pale green
            2: "#87CEEB",   # sky blue
            3: "#DDA0DD",   # plum
            4: "#F0E68C",   # khaki
            5: "#E6A8D7",   # light purple
            6: "#98D8D8",   # light cyan
            7: "#FFB347",   # pastel orange
            8: "#87AFC7"    # pastel blue
        }
        
        # Create density plot for each cluster
        for label in sorted(df['labels'].unique()):
            if label == -1:  # Skip noise points for density
                continue
                
            mask = df['labels'] == label
            points = df[mask]
            
            if len(points) >= 4:  # Need minimum points for KDE
                # Calculate kernel density estimate
                kde = gaussian_kde(points[['X', 'Y']].values.T)
                
                # Create a grid of points
                x_range = np.linspace(points['X'].min(), points['X'].max(), 100)
                y_range = np.linspace(points['Y'].min(), points['Y'].max(), 100)
                x_grid, y_grid = np.meshgrid(x_range, y_range)
                
                # Evaluate KDE on the grid
                positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                z = kde(positions).reshape(x_grid.shape)
                
                # Plot contour with transparency
                contours = ax.contourf(x_grid, y_grid, z, levels=10,
                                    cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                                        '', ['white', colors[label]]),
                                    alpha=0.3)
        
        # Plot points
        for label in sorted(df['labels'].unique()):
            mask = df['labels'] == label
            points = df[mask]
            
            if label == -1:  # Noise points
                ax.scatter(
                    points['X'],
                    points['Y'], 
                    c='gray', 
                    s=10,
                    alpha=0.2, 
                    label='Noise'
                )
            else:
                ax.scatter(
                    points['X'],
                    points['Y'], 
                    c=colors[label], 
                    s=20, 
                    alpha=0.6,
                    label=f'{self.cluster_summaries.get(label, f"Cluster {label}")}'
                )
        
        if style == "paper":
            # Publication-style formatting
            ax.grid(False)  # Remove grid for cleaner look
            ax.set_xlabel(f'{self.method.upper()} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{self.method.upper()} Dimension 2', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['bottom'].set_alpha(0.3)
            plt.title('Document Clustering Results', fontsize=14, pad=20)
            
            # Clean up legend
            legend = plt.legend(markerscale=0.8, frameon=True, 
                            fontsize=8, bbox_to_anchor=(1.05, 1), 
                            loc='upper left')
            legend.get_frame().set_alpha(0.8)
        else:
            ax.set_axis_off()
        
        plt.tight_layout()
        return fig

    def _show_plotly_enhanced(self, df, style="paper", show_summaries=True, include_noise_points=True):
        """Enhanced plotly visualization with soft, gradient-like cluster boundaries"""
        
        # Base color palette
        base_colors = {
            -1: "rgba(230, 230, 230, 0.5)",  # noise color
            0: "rgba(255, 182, 193, 0.7)",    # light pink
            1: "rgba(152, 251, 152, 0.7)",    # pale green
            2: "rgba(135, 206, 235, 0.7)",    # sky blue
            3: "rgba(221, 160, 221, 0.7)",    # plum
            4: "rgba(240, 230, 140, 0.7)",    # khaki
            5: "rgba(230, 168, 215, 0.7)",    # light purple
            6: "rgba(152, 216, 216, 0.7)",    # light cyan
            7: "rgba(255, 179, 71, 0.7)",     # pastel orange
            8: "rgba(135, 175, 199, 0.7)"     # pastel blue
        }
        
        # Dynamically generate colors for additional clusters
        def get_color(label):
            if label in base_colors:
                return base_colors[label]
            else:
                # Generate a random color for labels not in the base_colors dictionary
                import random
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                return f"rgba({r}, {g}, {b}, 0.7)"
        
        fig = go.Figure()
        
        # Add density contours for each cluster
        for label in sorted(df['labels'].unique()):
            if label == -1:  # Skip noise points for density
                continue
                
            mask = df['labels'] == label
            points = df[mask]
            
            if len(points) >= 4:  # Need minimum points for contours
                fig.add_trace(go.Histogram2dContour(
                    x=points['X'],
                    y=points['Y'],
                    colorscale=[[0, 'rgba(255,255,255,0)'], 
                            [1, get_color(label)]],
                    showscale=False,
                    ncontours=15,
                    contours=dict(coloring='fill'),
                    opacity=0.7,
                    name=f'Cluster {label} density'
                ))
        
        # Add scatter points
        for label in sorted(df['labels'].unique()):
            mask = df['labels'] == label
            points = df[mask]
            
            if label == -1:
                if include_noise_points:
                    # Noise points
                    fig.add_trace(go.Scatter(
                        x=points['X'], y=points['Y'],
                        mode='markers',
                        marker=dict(size=2, color='grey', opacity=0.1),
                        name='Noise',
                        hovertemplate='%{customdata}<extra></extra>',
                        customdata=points['content_display']
                    ))
            else:
                # Cluster points
                fig.add_trace(go.Scatter(
                    x=points['X'], y=points['Y'],
                    mode='markers',
                    marker=dict(size=2, color=get_color(label).replace('0.7', '1')),
                    name=f'{self.cluster_summaries.get(label, f"Cluster {label}")}' if show_summaries else f"Cluster {label}",
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=points['content_display']
                ))
        
        if style == "paper":
            # Publication-style layout
            fig.update_layout(
                template="plotly_white",
                width=1200,
                height=800,
                # title=dict(
                #     text='Clustering',
                #     x=0.5,
                #     y=0.95
                # ),
                xaxis=dict(
                    title=f'{self.method.upper()} Dimension 1',
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title=f'{self.method.upper()} Dimension 2',
                    showgrid=False,
                    zeroline=False
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bordercolor="Black",
                    borderwidth=1
                ),
                margin=dict(r=150)
            )
        else:
            fig.update_layout(
                template="plotly_white",
                width=1200,
                height=800,
                showlegend=True
            )
        
        fig.show()
        
        return fig

    def _show_mpl(self, df):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        df["color"] = df["labels"].apply(lambda x: "C0" if x==-1 else f"C{(x%9)+1}")

        df.plot(
            kind="scatter",
            x="X",
            y="Y",
            c="labels",
            s=0.75,
            alpha=0.8,
            linewidth=0,
            color=df["color"],
            ax=ax,
            colorbar=False,
        )
        if self.summary_create:
            for label in self.cluster_summaries.keys():
                if label == -1:
                    continue
                summary = self.cluster_summaries[label]
                position = self.cluster_centers[label]
                t= ax.text(
                    position[0],
                    position[1],
                    summary,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=4,
                )
                t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))
            ax.set_axis_off()

    

    def hex_to_rgba(self, hex_color, alpha=0.2):
        """Convert hex color to rgba string."""
        rgb = mcolors.hex2color(hex_color)  # Convert hex to RGB
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"

    def _show_plotly_with_boundaries(self, df):
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color=df["labels"].astype(str),  # Convert labels to string for categorical coloring
            hover_data={"content_display": True, "X": False, "Y": False},
            width=1600,
            height=800
        )

        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

        fig.update_traces(
            marker=dict(size=3, opacity=0.7),
            selector=dict(mode="markers"),
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Define a color palette for hulls
        colors = px.colors.qualitative.Plotly

        # Compute convex hulls for each cluster
        for i, label in enumerate(df["labels"].unique()):
            if label == -1:  # Ignore noise points
                continue

            cluster_points = df[df["labels"] == label][["X", "Y"]].values
            if len(cluster_points) < 3:  # ConvexHull needs at least 3 points
                continue

            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]

            # Close the hull shape
            hull_points = np.vstack([hull_points, hull_points[0]])

            fig.add_trace(go.Scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                fill="toself",
                fillcolor=self.hex_to_rgba(colors[i % len(colors)], 0.2),  # Use RGBA for transparency
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ))

        fig.show()



    def _show_plotly(self, df):
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="labels",
            hover_data={"content_display": True, "X": False, "Y": False},
            width=1600,
            height=800,
            color_continuous_scale="HSV",
        )

        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

        fig.update_traces(
            marker=dict(size=1, opacity=0.8),  # color="white"
            selector=dict(mode="markers"),
        )

        # Change template to "plotly_white" instead of "plotly_dark" for easier read
        fig.update_layout(
            template="plotly_dark",
            # # You can also explicitly set the background color:
            # paper_bgcolor='white',
            # plot_bgcolor='white'
        )

        # show cluster summaries
        if self.summary_create:
            for label in self.cluster_summaries.keys():
                if label == -1:
                    continue
                summary = self.cluster_summaries[label]
                position = self.cluster_centers[label]

                fig.add_annotation(
                    x=position[0],
                    y=position[1],
                    text=summary,
                    showarrow=False,
                    yshift=0,
                )

        fig.show()

    def _show_plotly_3d(self, df, style="paper"):
        """Enhanced 3D plotly visualization"""
        
        # Base color palette
        base_colors = {
            -1: "rgba(230, 230, 230, 0.5)",  # noise color
            0: "rgba(255, 182, 193, 0.7)",    # light pink
            1: "rgba(152, 251, 152, 0.7)",    # pale green
            2: "rgba(135, 206, 235, 0.7)",    # sky blue
            3: "rgba(221, 160, 221, 0.7)",    # plum
            4: "rgba(240, 230, 140, 0.7)",    # khaki
            5: "rgba(230, 168, 215, 0.7)",    # light purple
            6: "rgba(152, 216, 216, 0.7)",    # light cyan
            7: "rgba(255, 179, 71, 0.7)",     # pastel orange
            8: "rgba(135, 175, 199, 0.7)"     # pastel blue
        }
        
        # Create the 3D scatter plot
        fig = go.Figure()
        
        # Add points for each cluster
        for label in sorted(df['labels'].unique()):
            mask = df['labels'] == label
            points = df[mask]
            
            color = base_colors.get(label, base_colors[0])  # Default to first color if not found
            
            # Different styling for noise points
            if label == -1:
                marker_props = dict(
                    size=3,
                    color='grey',
                    opacity=0.1,
                    symbol='circle'
                )
                name = 'Noise'
            else:
                marker_props = dict(
                    size=4,
                    color=color,
                    opacity=0.7,
                    symbol='circle'
                )
                name = f'Cluster {label}'
                
                # Add cluster centers if summaries exist
                if hasattr(self, 'cluster_summaries') and self.cluster_summaries:
                    summary = self.cluster_summaries.get(label, '')
                    if summary and summary != 'None':
                        center_x = points['X'].mean()
                        center_y = points['Y'].mean()
                        center_z = points['Z'].mean()
                        
                        # Add text annotation for cluster center
                        fig.add_trace(go.Scatter3d(
                            x=[center_x],
                            y=[center_y],
                            z=[center_z],
                            mode='text',
                            text=[summary],
                            textposition='middle center',
                            textfont=dict(size=10, color='black'),
                            showlegend=False
                        ))
            
            # Add the scatter points
            fig.add_trace(go.Scatter3d(
                x=points['X'],
                y=points['Y'],
                z=points['Z'],
                mode='markers',
                marker=marker_props,
                name=name,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=points['content_display']
            ))
        
        # Update layout based on style
        if style == "paper":
            fig.update_layout(
                template="plotly_white",
                width=1200,
                height=800,
                title=dict(
                    text=f'3D {self.method.upper()} Clustering Visualization',
                    x=0.5,
                    y=0.95
                ),
                scene=dict(
                    xaxis_title=f'{self.method.upper()} Dimension 1',
                    yaxis_title=f'{self.method.upper()} Dimension 2',
                    zaxis_title=f'{self.method.upper()} Dimension 3',
                    xaxis=dict(showgrid=True, zeroline=False, showline=True),
                    yaxis=dict(showgrid=True, zeroline=False, showline=True),
                    zaxis=dict(showgrid=True, zeroline=False, showline=True),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bordercolor="Black",
                    borderwidth=1
                ),
                margin=dict(r=150)
            )
        else:
            fig.update_layout(
                template="plotly_white",
                width=1200,
                height=800,
                showlegend=True,
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            )
        
        fig.show()
        
        return fig