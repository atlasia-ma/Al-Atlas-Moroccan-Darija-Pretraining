from clustering import ClusterClassifier
import gc
import torch
from datasets import load_dataset
from cycler import cycler
import matplotlib.pyplot as plt
from cycler import cycler
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")

if __name__ == '__main__':
    print("[INFO] Cuda Empty Cashe...")
    gc.collect()
    torch.cuda.empty_cache()

    print("[INFO] Load Dataset...")
    ds=load_dataset("atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset",split="train",token=HF_TOKEN)

    print("[INFO] Load Cluster Classifier...")
    cluster_classifier=ClusterClassifier(
        embed_model_name="BAAI/bge-m3",
        embed_device="cuda",
        use_gemini=True,
        gemini_token=GEMINI_TOKEN,
        embed_batch_size=768
    )

    print("[INFO] Fit Cluster Classifier...")
    embs,labels,summaries=cluster_classifier.fit(ds["text"])

    print("[INFO] Save Cluster Classifier...")
    cluster_classifier.save("cluster_classifier")

    print("[INFO] Show Cluster Classifier Result...")
    default_cycler = (cycler(color=[
        "#0F0A0A",  # Dark Grayish Black
        "#FF6600",  # Vivid Orange
        "#FFBE00",  # Bright Yellow
        "#496767",  # Muted Teal
        "#87A19E",  # Soft Cyan-Gray
        "#FF9200",  # Deep Orange
        "#0F3538",  # Dark Cyan-Teal
        "#F8E08E",  # Pastel Yellow
        "#0F2021",  # Very Dark Cyan
        "#FAFAF0",  # Off-White
        "#8B0000",  # Dark Red
        "#4682B4",  # Steel Blue
        "#32CD32",  # Lime Green
        "#9370DB",  # Medium Purple
        "#FFD700",  # Gold
        "#DC143C",  # Crimson Red
        "#00CED1",  # Dark Turquoise
        "#2E8B57",  # Sea Green
        "#FF1493"   # Deep Pink
        ]))
    plt.rc('axes', prop_cycle=default_cycler)
    cluster_classifier.show(True)