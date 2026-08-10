import csv
import numpy as np
import umap
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def load_embedding_file(path, dtype=np.float32):
    """Load a node2vec-style embedding file.

    First line: "n_nodes dim"
    Each subsequent line: "node_id v1 v2 ... v_dim"
    """
    node_ids = []
    vectors = []

    with open(path, "r") as f:
        first_line = f.readline().strip()
        n_nodes, dim = map(int, first_line.split())

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            node_id = parts[0]
            vec = np.array(parts[1:], dtype=dtype)

            if vec.shape[0] != dim:
                raise ValueError(f"Dimension mismatch for node {node_id}")

            node_ids.append(node_id)
            vectors.append(vec)

    X = np.vstack(vectors)
    return X, node_ids


def load_id_to_label_map(path):
    """Load a CSV with columns: label,node_id (header required), e.g. USA,0.

    Returns a dict[str, str] mapping node_id -> label (e.g. country code).
    """
    mapping = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row
        for row in reader:
            if not row:
                continue
            label, node_id = row[0].strip(), row[1].strip()
            mapping[node_id] = label
    return mapping


def load_category_map(path):
    """Load an optional CSV with columns: label,category (e.g. country_code,continent).

    Used only for coloring points; independent of the id->label map.
    """
    mapping = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            label, category = row[0].strip(), row[1].strip()
            mapping[label] = category
    return mapping


def plot_umap(
    X,
    labels=None,
    categories=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    title="UMAP projection",
    output=None,
    random_state=42,
    annotate=True,
):
    """
    Generate and plot a 2D UMAP projection.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    labels : list[str], optional
        Text labels to annotate each point (e.g. country codes)
    categories : list[str], optional
        Category per point (e.g. continent) used only for coloring
    annotate : bool
        Whether to draw text labels next to points
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(9, 7))

    if categories is not None:
        # Map category strings -> integer codes for coloring
        unique_cats = sorted(set(categories))
        cat_to_code = {c: i for i, c in enumerate(unique_cats)}
        codes = np.array([cat_to_code[c] for c in categories])

        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=codes, cmap="tab20", s=25, edgecolors="k", linewidths=0.3
        )
        handles, _ = scatter.legend_elements(num=len(unique_cats))
        plt.legend(handles, unique_cats, title="Category",
                   bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=25, edgecolors="k", linewidths=0.3)

    if annotate and labels is not None:
        texts = []
        for i, node in enumerate(labels):
            texts.append(
                plt.annotate(str(node), (X_2d[i, 0], X_2d[i, 1]), fontsize=7)
            )

        if HAS_ADJUST_TEXT and texts:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        elif not HAS_ADJUST_TEXT:
            print(
                "Note: install 'adjustText' (pip install adjustText) "
                "to automatically de-overlap labels."
            )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.show()

    return X_2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                         help="Path to node2vec embedding file (text format)")
    parser.add_argument("--id_map", type=str, default=None,
                         help="CSV mapping node_id -> label, e.g. country code")
    parser.add_argument("--category_map", type=str, default=None,
                         help="CSV mapping label -> category, e.g. country_code -> continent (for coloring)")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--no_annotate", action="store_true",
                         help="Disable point text labels (useful for very dense graphs)")
    parser.add_argument("--output", type=str, default=None,
                         help="If provided, save the figure to this path")

    args = parser.parse_args()

    X, node_ids = load_embedding_file(args.input)

    if args.id_map:
        id_map = load_id_to_label_map(args.id_map)
        display_labels = [id_map.get(nid, nid) for nid in node_ids]
    else:
        display_labels = node_ids

    categories = None
    if args.category_map:
        cat_map = load_category_map(args.category_map)
        categories = [cat_map.get(lbl, "Unknown") for lbl in display_labels]

    plot_umap(
        X,
        labels=display_labels,
        categories=categories,
        metric=args.metric,
        annotate=not args.no_annotate,
        output=args.output,
    )