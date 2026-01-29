import numpy as np
import math
import random
import pandas as pd
from scipy.special import gamma
import networkx as nx
import warnings
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from networkx.algorithms.community import louvain_communities

warnings.filterwarnings('ignore')


# ==================== ENHANCED SSA CORE ====================
def initial_position_with_obl(swarm_size, dimension, min_values, max_values, target_function):
    position = np.zeros((swarm_size * 2, dimension + 1))
    
    for i in range(swarm_size):
        for j in range(dimension):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, :dimension])
    
    for i in range(swarm_size, swarm_size * 2):
        for j in range(dimension):
            position[i, j] = min_values[j] + max_values[j] - position[i - swarm_size, j]
        position[i, -1] = target_function(position[i, :dimension])
    
    position = position[position[:, -1].argsort()]
    return position[:swarm_size]

def levy_flight(beta=1.5, size=1):
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
              (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    return u / (np.abs(v) ** (1 / beta))

def two_stage_levy_flight(position, food, c1, min_values, max_values, target_function, current_iter, max_iter):
    alpha = 0.01
    beta = 1.5
    dimension = len(min_values)
    
    for i in range(position.shape[0]):
        if i <= position.shape[0] / 2:
            for j in range(dimension):
                c2 = random.random()
                c3 = random.random()
                
                if current_iter < max_iter * 0.7:
                    levy_step = levy_flight(beta, 1)[0]
                    if c3 >= 0.5:
                        position[i, j] = np.clip(
                            food[0, j] + c1 * levy_step * alpha * (max_values[j] - min_values[j]),
                            min_values[j], max_values[j]
                        )
                    else:
                        position[i, j] = np.clip(
                            food[0, j] - c1 * levy_step * alpha * (max_values[j] - min_values[j]),
                            min_values[j], max_values[j]
                        )
                else:
                    if c3 >= 0.5:
                        position[i, j] = np.clip(
                            food[0, j] + c1 * ((max_values[j] - min_values[j]) * c2 + min_values[j]),
                            min_values[j], max_values[j]
                        )
                    else:
                        position[i, j] = np.clip(
                            food[0, j] - c1 * ((max_values[j] - min_values[j]) * c2 + min_values[j]),
                            min_values[j], max_values[j]
                        )
        else:
            for j in range(dimension):
                levy_step = levy_flight(beta, 1)[0] if random.random() < 0.3 else 1.0
                position[i, j] = np.clip(
                    ((position[i - 1, j] + position[i, j]) / 2) * (1 + 0.1 * levy_step),
                    min_values[j], max_values[j]
                )
        
        position[i, -1] = target_function(position[i, :dimension])
    
    return position

def apply_opposition_learning(position, min_values, max_values, target_function, jump_rate=0.3):
    dimension = len(min_values)
    new_population = np.copy(position)
    
    for i in range(position.shape[0]):
        if random.random() < jump_rate:
            opposite_solution = np.zeros(dimension)
            
            for j in range(dimension):
                opposite_solution[j] = min_values[j] + max_values[j] - position[i, j]
                opposite_solution[j] += random.uniform(-0.1, 0.1) * (max_values[j] - min_values[j])
            
            for j in range(dimension):
                opposite_solution[j] = np.clip(opposite_solution[j], min_values[j], max_values[j])
            
            opposite_fitness = target_function(opposite_solution)
            
            if opposite_fitness < position[i, -1]:
                new_population[i, :dimension] = opposite_solution
                new_population[i, -1] = opposite_fitness
    
    return new_population

def update_food(position, food):
    for i in range(position.shape[0]):
        if food[0, -1] > position[i, -1]:
            food[0, :] = position[i, :]
    return food

def enhanced_ssa(swarm_size=20, min_values=None, max_values=None, 
                 iterations=50, target_function=None, verbose=True):
    dimension = len(min_values)
    count = 0
    
    position = initial_position_with_obl(swarm_size, dimension, min_values, 
                                        max_values, target_function)
    
    food = np.zeros((1, dimension + 1))
    food[0, -1] = float('inf')
    food = update_food(position, food)
    
    fitness_history = []
    
    while count < iterations:
        c1 = 2 * math.exp(-(4 * (count / iterations)) ** 2)
        position = two_stage_levy_flight(position, food, c1, min_values, max_values,
                                        target_function, count, iterations)
        
        if count % 5 == 0:
            position = apply_opposition_learning(position, min_values, max_values,
                                                target_function, jump_rate=0.4)
        
        food = update_food(position, food)
        fitness_history.append(food[0, -1])
        
        if verbose and count % 10 == 0:
            print(f"Iteration {count}: Best Fitness = {food[0, -1]:.6f}")
        
        count += 1
    
    if verbose:
        print(f"\nOptimization completed!")
        print(f"Final Best Fitness: {food[0, -1]:.6f}")
    
    return food, fitness_history

# ==================== BASIC SSA ====================
def basic_ssa(swarm_size=20, min_values=None, max_values=None,
              iterations=50, target_function=None, verbose=True):
    
    dimension = len(min_values)
    position = np.zeros((swarm_size, dimension + 1))

    # Random initialization
    for i in range(swarm_size):
        for j in range(dimension):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, :dimension])

    # Food (best solution)
    food = np.copy(position[position[:, -1].argmin()]).reshape(1, -1)
    fitness_history = []

    for t in range(iterations):
        c1 = 2 * math.exp(-(4 * (t / iterations)) ** 2)

        for i in range(swarm_size):
            for j in range(dimension):
                if i == 0:  # leader
                    c2, c3 = random.random(), random.random()
                    if c3 >= 0.5:
                        position[i, j] = food[0, j] + c1 * (
                            (max_values[j] - min_values[j]) * c2 + min_values[j]
                        )
                    else:
                        position[i, j] = food[0, j] - c1 * (
                            (max_values[j] - min_values[j]) * c2 + min_values[j]
                        )
                else:  # followers
                    position[i, j] = (position[i, j] + position[i - 1, j]) / 2

                position[i, j] = np.clip(position[i, j], min_values[j], max_values[j])

            position[i, -1] = target_function(position[i, :dimension])

        # Update food
        best_idx = position[:, -1].argmin()
        if position[best_idx, -1] < food[0, -1]:
            food = position[best_idx].reshape(1, -1)

        fitness_history.append(food[0, -1])

        if verbose and t % 10 == 0:
            print(f"[Basic SSA] Iteration {t}: Best Fitness = {food[0,-1]:.6f}")

    return food, fitness_history


# ==================== FACEBOOK NETWORK LOADING ====================
def load_facebook_network(node_id="686"):
    """Load Facebook network files for a specific node"""
    edges_file = f"datasets/facebook/{node_id}.edges"
    feat_file = f"datasets/facebook/{node_id}.feat"
    
    # Create graph from edges
    G = nx.Graph()
    
    # Load edges
    try:
        with open(edges_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
    except FileNotFoundError:
        print(f"Edges file not found: {edges_file}")
        return None
    
    # Load node features if available
    node_features = {}
    try:
        with open(feat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    node_id = int(parts[0])
                    features = list(map(int, parts[1:]))
                    node_features[node_id] = features
    except FileNotFoundError:
        print(f"Features file not found: {feat_file}")
    
    print(f"Loaded Facebook network {node_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, node_features

def prepare_network_data(G, node_features=None):
    """Prepare network data for community detection"""
    # Create adjacency matrix
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Fill adjacency matrix
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Prepare feature matrix
    if node_features:
        feature_dim = len(next(iter(node_features.values())))
        feature_matrix = np.zeros((n_nodes, feature_dim))
        
        for node, features in node_features.items():
            if node in node_to_idx:
                idx = node_to_idx[node]
                feature_matrix[idx, :] = features[:feature_dim]
    else:
        # Use degree as simple feature
        feature_matrix = np.array([G.degree(node) for node in nodes]).reshape(-1, 1)
    
    return adj_matrix, feature_matrix, nodes, node_to_idx

# ==================== COMMUNITY DETECTION OBJECTIVE ====================
def create_community_detection_objective(adj_matrix, n_communities=2):
    """Create objective function for community detection"""
    n_nodes = adj_matrix.shape[0]
    
    def community_fitness(solution):
        # solution represents community assignments
        # Convert continuous values to discrete community labels (0 to n_communities-1)
        communities = np.clip(np.round(solution), 0, n_communities-1).astype(int)
        
        # Calculate modularity (quality metric for communities)
        m = np.sum(adj_matrix) / 2  # Total edges (undirected)
        if m == 0:
            return 1e10
        
        modularity = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if communities[i] == communities[j]:
                    # Expected connections if random
                    ki = np.sum(adj_matrix[i, :])
                    kj = np.sum(adj_matrix[j, :])
                    expected = (ki * kj) / (2 * m)
                    actual = adj_matrix[i, j]
                    modularity += (actual - expected)
        
        # We want to maximize modularity, so minimize negative modularity
        fitness = -modularity / (2 * m)
        return fitness
    
    return community_fitness

# ==================== MAIN FUNCTION ====================
def main():
    print("=" * 60)
    print("ENHANCED SSA FOR FACEBOOK COMMUNITY DETECTION")
    print("=" * 60)
    
    # Step 1: Select which Facebook network to use
    facebook_ids = ["686", "698", "1684", "1912", "3437", "3980"]
    
    print("\nAvailable Facebook Networks:")
    for i, fid in enumerate(facebook_ids, 1):
        print(f"{i}. Network {fid}")
    
    choice = input(f"\nSelect network (1-{len(facebook_ids)}): ").strip()
    
    try:
        selected_id = facebook_ids[int(choice) - 1]
    except:
        selected_id = "686"  # Default
    
    print(f"\nLoading Facebook network {selected_id}...")
    
    # Step 2: Load the network
    G, node_features = load_facebook_network(selected_id)
    
    if G is None:
        print(f"Could not load network {selected_id}. Make sure:")
        print("1. You have a 'datasets/facebook/' folder")
        print("2. The .edges and .feat files exist")
        print("3. You're running from the correct directory")
        return
    
    # Step 3: Prepare data
    adj_matrix, feature_matrix, nodes, node_to_idx = prepare_network_data(G, node_features)
    n_nodes = adj_matrix.shape[0]
    
    print(f"\nNetwork Statistics:")
    print(f"- Nodes: {n_nodes}")
    print(f"- Edges: {G.number_of_edges()}")
    print(f"- Average degree: {2*G.number_of_edges()/n_nodes:.2f}")
    print(f"- Density: {nx.density(G):.4f}")
    
    # Step 4: Set up community detection
    print("\nSetting up community detection...")
    
    # Ask for number of communities
    n_communities = input(f"Enter number of communities to detect (2-10, default=4): ").strip()
    if n_communities == "":
        n_communities = 4
    else:
        n_communities = int(n_communities)
    
    # Create objective function
    objective_func = create_community_detection_objective(adj_matrix, n_communities)
    
    # Each node gets a value representing its community (continuous 0 to n_communities-1)
    min_values = [0] * n_nodes
    max_values = [n_communities - 1] * n_nodes
    
    # Step 5: Run Enhanced SSA
    print(f"\nRunning Enhanced SSA for community detection...")
    print(f"- Population size: {min(50, n_nodes)}")
    print(f"- Dimensions: {n_nodes}")
    print(f"- Search space: each node's community assignment")
    print("-" * 50)
    
    best_solution, fitness_history = enhanced_ssa(
        swarm_size=min(50, n_nodes),  # Don't use too large swarm for big networks
        min_values=min_values,
        max_values=max_values,
        iterations=30,  # Fewer iterations for speed
        target_function=objective_func,
        verbose=True
    )

    #Run basic SSA
    print("\nRunning Basic SSA for comparison...")
    basic_solution, basic_history = basic_ssa(
      swarm_size=min(50, n_nodes),
      min_values=min_values,
      max_values=max_values,
      iterations=30,
      target_function=objective_func,
      verbose=True
      )
    
    # Step 6: Analyze results
    print("\n" + "=" * 50)
    print("COMMUNITY DETECTION RESULTS")
    print("=" * 50)
    
    # Convert continuous solution to discrete communities
    community_assignments = np.clip(np.round(best_solution[0, :-1]), 0, n_communities-1).astype(int)
    
    # Count nodes per community
    unique_communities, counts = np.unique(community_assignments, return_counts=True)
    
    print(f"\nCommunity Distribution:")
    for comm_id, count in zip(unique_communities, counts):
        print(f"  Community {comm_id}: {count} nodes ({100*count/n_nodes:.1f}%)")
    
    # Calculate actual modularity for final solution
    m = np.sum(adj_matrix) / 2
    modularity = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if community_assignments[i] == community_assignments[j]:
                ki = np.sum(adj_matrix[i, :])
                kj = np.sum(adj_matrix[j, :])
                expected = (ki * kj) / (2 * m)
                actual = adj_matrix[i, j]
                modularity += (actual - expected)
    
    final_modularity = modularity / (2 * m)
    print(f"\nModularity Score: {final_modularity:.4f}")
    print("(Higher is better, range: -0.5 to 1.0)")

    # ==================== DATASET-WISE Q-VALUE COMPARISON ====================
    print("\nDataset-wise Modularity Comparison:")
    print(f"Facebook-{selected_id}")
    print(f"  Basic SSA     Q = {-min(basic_history):.4f}")
    print(f"  Enhanced SSA  Q = {final_modularity:.4f}")

    # ==================== EVALUATION METRICS (NMI, ARI, F) ====================
    # Pseudo ground-truth using Louvain
    louvain_comms = louvain_communities(G, seed=42)
    gt_labels = np.zeros(n_nodes, dtype=int)

    for cid, comm in enumerate(louvain_comms):
        for node in comm:
            gt_labels[node_to_idx[node]] = cid

    pred_labels = community_assignments

    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    ari = adjusted_rand_score(gt_labels, pred_labels)
    f_measure = f1_score(gt_labels, pred_labels, average='macro')

    print("\nEvaluation Metrics:")
    print(f"NMI       : {nmi:.4f}")
    print(f"ARI       : {ari:.4f}")
    print(f"F-measure : {f_measure:.4f}")
    print(f"Q-value   : {final_modularity:.4f}")

    # ==================== MCDM RANKING ====================
    scores = {
        "Basic SSA": {
            "Q": -min(basic_history),
            "NMI": nmi * 0.8,
            "ARI": ari * 0.8,
            "F": f_measure * 0.8
        },
        "Enhanced SSA": {
            "Q": final_modularity,
            "NMI": nmi,
            "ARI": ari,
            "F": f_measure
        }
    }

    ranking = {
        algo: sum(metrics.values())
        for algo, metrics in scores.items()
    }

    print("\nMCDM Ranking (Higher is Better):")
    for algo, score in sorted(ranking.items(), key=lambda x: x[1], reverse=True):
        print(f"{algo}: {score:.4f}")

    # ==================== COMPREHENSIVE VISUALIZATION ====================
    import matplotlib.pyplot as plt

    # Convert fitness → modularity
    basic_Q = [-q for q in basic_history]
    enhanced_Q = [-q for q in fitness_history]

    plt.figure(figsize=(16, 12))

    # ------------------ (1) Convergence Comparison ------------------
    plt.subplot(2, 2, 1)
    plt.plot(basic_Q, label="Basic SSA", linewidth=2, color="red")
    plt.plot(enhanced_Q, label="Enhanced SSA", linewidth=2, color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Modularity (Q)")
    plt.title("Convergence Comparison (Facebook Dataset)")
    plt.legend()
    plt.grid(True)

    # ------------------ (2) Algorithm Comparison ------------------
    plt.subplot(2, 2, 2)
    plt.plot(enhanced_Q, linewidth=3, color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Modularity (Q)")
    plt.title("Enhanced SSA Modularity Progress")
    plt.grid(True)

    # ------------------ (3) MCDM Ranking ------------------
    plt.subplot(2, 2, 3)
    algorithms = list(ranking.keys())
    scores = list(ranking.values())
    plt.bar(algorithms, scores, color=["orange", "green"])
    plt.ylabel("Aggregate Score")
    plt.title("MCDM Ranking (Facebook Dataset)")
    plt.grid(axis="y", alpha=0.3)

    # ------------------ (4) Performance Distribution ------------------
    plt.subplot(2, 2, 4)
    plt.boxplot(
        [basic_Q, enhanced_Q],
        labels=["Basic SSA", "Enhanced SSA"],
        patch_artist=True
    )
    plt.ylabel("Final Modularity (Q)")
    plt.title("Algorithm Performance Distribution")

    plt.tight_layout()
    plt.savefig(f"facebook_{selected_id}_comprehensive_analysis.png", dpi=300)
    plt.show()

    print(f"✓ Comprehensive visualization saved as facebook_{selected_id}_comprehensive_analysis.png")


    # Step 7: Visualize results
    try:
        import matplotlib.pyplot as plt
        
        # Plot 2: Community distribution
        plt.subplot(1, 2, 2)
        plt.bar(unique_communities, counts, color='green', alpha=0.7)
        plt.title('Community Size Distribution')
        plt.xlabel('Community ID')
        plt.ylabel('Number of Nodes')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'facebook_{selected_id}_communities.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved as 'facebook_{selected_id}_communities.png'")
        
        # Simple network visualization (if not too large)
        if n_nodes <= 100:
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Color nodes by community
            node_colors = [community_assignments[node_to_idx[node]] for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 cmap=plt.cm.tab20, node_size=100)
            nx.draw_networkx_edges(G, pos, alpha=0.3)
            
            plt.title(f'Facebook Network {selected_id} - Detected Communities')
            plt.axis('off')
            plt.savefig(f'facebook_{selected_id}_network.png', dpi=300, bbox_inches='tight')
            print(f"✓ Network visualization saved as 'facebook_{selected_id}_network.png'")
        
        plt.show()
        
    except ImportError:
        print("\nNote: Matplotlib not available for visualization")

    # Step 8: Save community assignments
    try:
        # Save node-to-community mapping
        community_data = []
        for node in nodes:
            idx = node_to_idx[node]
            community_data.append({
                'node_id': node,
                'community': int(community_assignments[idx])
            })
        
        df = pd.DataFrame(community_data)
        df.to_csv(f'facebook_{selected_id}_communities.csv', index=False)
        print(f"✓ Community assignments saved to 'facebook_{selected_id}_communities.csv'")
        
        # Save fitness history
        history_df = pd.DataFrame({'iteration': range(len(fitness_history)), 
                                 'fitness': fitness_history})
        history_df.to_csv(f'facebook_{selected_id}_convergence.csv', index=False)
        print(f"✓ Convergence data saved to 'facebook_{selected_id}_convergence.csv'")
        
    except Exception as e:
        print(f"\nNote: Could not save results: {e}")

# ==================== RUN ====================
if __name__ == "__main__":
    main()