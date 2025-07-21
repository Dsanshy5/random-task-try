import fitz

def extract_candidates(pdf_path):
    doc = fitz.open(pdf_path)
    candidates = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text.split()) > 12:
                        continue
                    
                    font_size = span["size"]
                    is_bold = "Bold" in span["font"] or "bold" in span["font"].lower()
                    font_weight = 1 if is_bold else 0

                    candidates.append({
                        "text": text,
                        "size": font_size,
                        "bold": font_weight,
                        "page": page_num
                    })

    return candidates

def cluster_headings(candidates, n_clusters=3):
    features = np.array([[c["size"], c["bold"]] for c in candidates])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)

    # Sort clusters by mean font size descending -> assign H1 > H2 > H3
    cluster_font_sizes = {
        i: features[kmeans.labels_ == i][:, 0].mean() for i in range(n_clusters)
    }
    sorted_clusters = sorted(cluster_font_sizes.items(), key=lambda x: -x[1])
    cluster_to_level = {cluster: f"H{i+1}" for i, (cluster, _) in enumerate(sorted_clusters)}

    headings = []
    for i, candidate in enumerate(candidates):
        cluster = kmeans.labels_[i]
        headings.append({
            "level": cluster_to_level[cluster],
            "text": candidate["text"],
            "page": candidate["page"]
        })

    return headings