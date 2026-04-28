import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt


def membership_inference(model, clients_dir: str) -> dict:
    """
    Embedding-based membership inference using cosine similarity.

    Intra-class similarity vs inter-class similarity. Members should
    have higher intra-class similarity (same person) than inter-class.
    """
    model.eval()
    all_embeddings = {}  # person_name -> list of numpy embeddings

    for client_folder in os.listdir(clients_dir):
        client_path = os.path.join(clients_dir, client_folder)
        if not os.path.isdir(client_path):
            continue
        for person in os.listdir(client_path):
            person_path = os.path.join(client_path, person)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for f in os.listdir(person_path):
                if f.endswith('.pt'):
                    tensor = torch.load(os.path.join(person_path, f), map_location='cpu')
                    if len(tensor.shape) == 3:
                        tensor = tensor.unsqueeze(0)
                    with torch.no_grad():
                        emb = model(tensor).squeeze().cpu().numpy()
                    # L2-normalize to get cosine similarity via dot product
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)
            if embeddings:
                all_embeddings[person] = embeddings

    intra_similarities = []
    for person, embeds in all_embeddings.items():
        if len(embeds) >= 2:
            for i in range(len(embeds)):
                for j in range(i+1, len(embeds)):
                    sim = float(np.dot(embeds[i], embeds[j]))
                    intra_similarities.append(sim)

    inter_similarities = []
    people = list(all_embeddings.keys())
    for i in range(len(people)):
        for j in range(i+1, len(people)):
            e1 = all_embeddings[people[i]][0]
            e2 = all_embeddings[people[j]][0]
            sim = float(np.dot(e1, e2))
            inter_similarities.append(sim)

    intra_mean = float(np.mean(intra_similarities)) if intra_similarities else 0.0
    inter_mean = float(np.mean(inter_similarities)) if inter_similarities else 0.0
    separation = intra_mean - inter_mean

    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "separation": separation,
        "intra_similarities": intra_similarities,
        "inter_similarities": inter_similarities,
    }


def evaluate_membership_inference(
    model,
    clients_dir: str,
    held_out_dir: str = None
) -> dict:
    """
    Evaluate membership inference attack on one model.
    """
    model.eval()
    
    member_distances = []
    non_member_distances = []
    
    # Pre-compute templates for each client (member)
    client_templates = {}
    
    for client_id in os.listdir(clients_dir):
        client_path = os.path.join(clients_dir, client_id)
        if not os.path.isdir(client_path):
            continue
            
        # Find person directory inside client
        people_dirs = [d for d in os.listdir(client_path) if os.path.isdir(os.path.join(client_path, d))]
        if not people_dirs:
            continue
            
        person_name = people_dirs[0]
        person_dir = os.path.join(client_path, person_name)
        
        # Load all tensors
        tensor_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(".pt")]
        if not tensor_files:
            continue
            
        tensors = []
        for tf in tensor_files:
            tensor = torch.load(tf, map_location="cpu")
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
            
        if len(tensors) < 2:
            continue
            
        # Split 80% train / 20% test
        split_idx = int(0.8 * len(tensors))
        train_tensors = tensors[:split_idx]
        test_tensors = tensors[split_idx:]
        
        with torch.no_grad():
            # Mean embedding of training
            train_embs = [model(t) for t in train_tensors]
            template = torch.mean(torch.stack(train_embs), dim=0).squeeze()
            client_templates[person_name] = template
            
            # Distance of tests to template
            for t in test_tensors:
                emb = model(t).squeeze()
                dist = float(torch.norm(emb - template))
                member_distances.append(dist)
                
    if held_out_dir and os.path.exists(held_out_dir):
        # We have real non-members
        pass # Optional, assuming None based on prompt's alternative
    else:
        # Simulate non-members by cross-comparing
        # Person A's test tensors compared to Person B's template
        # Re-iterate and cross compare
        for client_id in os.listdir(clients_dir):
            client_path = os.path.join(clients_dir, client_id)
            if not os.path.isdir(client_path): continue
            
            people_dirs = [d for d in os.listdir(client_path) if os.path.isdir(os.path.join(client_path, d))]
            if not people_dirs: continue
            
            person_name = people_dirs[0]
            person_dir = os.path.join(client_path, person_name)
            
            tensor_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(".pt")]
            if len(tensor_files) < 2: continue
            
            split_idx = int(0.8 * len(tensor_files))
            test_files = tensor_files[split_idx:]
            
            for tf in test_files:
                tensor = torch.load(tf, map_location="cpu")
                if len(tensor.shape) == 3: tensor = tensor.unsqueeze(0)
                
                with torch.no_grad():
                    emb = model(tensor).squeeze()
                    
                # Find nearest template that is NOT this person
                min_dist = float('inf')
                for other_name, template in client_templates.items():
                    if other_name == person_name: continue
                    dist = float(torch.norm(emb - template))
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist != float('inf'):
                    non_member_distances.append(min_dist)

    member_mean = float(np.mean(member_distances)) if member_distances else 0.0
    non_member_mean = float(np.mean(non_member_distances)) if non_member_distances else 0.0
    gap = non_member_mean - member_mean
    advantage = gap / (member_mean + non_member_mean) if (member_mean + non_member_mean) > 0 else 0.0
    
    return {
        "member_distances": member_distances,
        "non_member_distances": non_member_distances,
        "member_mean": member_mean,
        "non_member_mean": non_member_mean,
        "gap": gap,
        "advantage": advantage
    }

def compare_membership_inference(
    model_no_dp,
    model_with_dp,
    clients_dir: str,
    output_path: str
) -> dict:
    
    res_a = membership_inference(model_no_dp, clients_dir)
    res_b = membership_inference(model_with_dp, clients_dir)
    
    # Interpretation text
    reduction = 0
    if res_a['advantage'] > 0:
        reduction = (res_a['advantage'] - res_b['advantage']) / res_a['advantage'] * 100
        
    interpretation = f"DP reduced membership inference advantage from {res_a['advantage']:.2f} to {res_b['advantage']:.2f} ({reduction:.0f}% reduction)"
    
    results = {
        "version_a": res_a,
        "version_b": res_b,
        "interpretation": interpretation
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def plot_membership_results(results: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Version A (No DP)', 'Version B (With DP)']
    member_means = [results['version_a']['member_mean'], results['version_b']['member_mean']]
    non_member_means = [results['version_a']['non_member_mean'], results['version_b']['non_member_mean']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, member_means, width, label='Members (Training Data)', color='blue', alpha=0.7)
    ax.bar(x + width/2, non_member_means, width, label='Non-Members (Held-out)', color='red', alpha=0.7)
    
    ax.set_ylabel('Mean Embedding Distance')
    ax.set_title('Membership Inference Gap (Smaller gap = Better privacy)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
