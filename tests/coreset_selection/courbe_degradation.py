checkpoints = [50, 100, 200, 500, 1000, len(X)]

results = []
for k in checkpoints:
    # Coreset déjà sélectionné par greedy — on prend les k premiers
    coreset_ids = original_indices[selected_idx[:k]]
    
    # Entraîner le transformer sur ce coreset
    model = train(coreset_ids)
    
    # Évaluer
    f1 = evaluate(model)
    sw = compute_sw(X[selected_idx[:k]], X)
    
    results.append((k, sw, f1))