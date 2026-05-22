def bredth_first_search(graph, start, goal):
    visited = set()  # To keep track of visited nodes
    queue = [start]  # Initialize the queue with the starting node

    while queue:
        node = queue.pop(0)  # Dequeue a node
        if node not in visited:
            print(node)  # Process the node (e.g., print it)
            if node == goal:
                return node
            visited.add(node)  # Mark the node as visited
            # Enqueue all unvisited neighbors
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return None
                    
# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}


print("Breadth-First Search from 'A' to 'E':")
print("Result:", bredth_first_search(graph, 'A', 'E'))


def depth_first_search(graph, start, goal, visited=None):
    if visited is None:
        visited = set()  # To keep track of visited nodes

    if start not in visited:
        print(start)  # Process the node (e.g., print it)
        if start == goal:
            return start
        visited.add(start)  # Mark the node as visited
        # Recursively visit all unvisited neighbors
        for neighbor in graph[start]:
            result = depth_first_search(graph, neighbor, goal, visited)
            if result is not None:
                return result
    return None
# Example usage:
print("\nDepth-First Search from 'A' to 'E':")
print("Result:", depth_first_search(graph, 'A', 'E'))


def greedy_best_first_search(graph, start, goal, heuristic):
    visited = set()  # To keep track of visited nodes
    queue = [(heuristic[start], start)]  # Initialize the queue with the starting node and its heuristic value

    while queue:
        _, node = queue.pop(0)  # Dequeue the node with the lowest heuristic value
        if node not in visited:
            print(node)  # Process the node (e.g., print it)
            visited.add(node)  # Mark the node as visited
            if node == goal:
                return node
            # Enqueue all unvisited neighbors with their heuristic values
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((heuristic[neighbor], neighbor))
                    queue.sort()  # Sort the queue based on heuristic values
    return None
# Example usage:
heuristic = {
    'A': 3,
    'B': 2,
    'C': 4,
    'D': 1,
    'E': 0,
    'F': 5
}

print("\nGreedy Best-First Search from 'A' to 'E':")
print("Result:", greedy_best_first_search(graph, 'A', 'E', heuristic))

def uniform_cost_search(graph, start, goal):
    visited = set()  # To keep track of visited nodes
    queue = [(0, start)]  # Initialize the queue with the starting node and its cost

    while queue:
        cost, node = queue.pop(0)  # Dequeue the node with the lowest cost
        if node not in visited:
            print(node)  # Process the node (e.g., print it)
            visited.add(node)  # Mark the node as visited
            if node == goal:
                return node
            # Enqueue all unvisited neighbors with their cumulative costs
            for neighbor, edge_cost in graph[node]:
                if neighbor not in visited:
                    queue.append((cost + edge_cost, neighbor))
                    queue.sort()  # Sort the queue based on cumulative costs
    return None
# Example usage:
weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('D', 2), ('E', 5)],
    'C': [('A', 4), ('F', 3)],
    'D': [('B', 2)],
    'E': [('B', 5), ('F', 1)],
    'F': [('C', 3), ('E', 1)]
}

print("\nUniform Cost Search from 'A' to 'E':")
print("Result:", uniform_cost_search(weighted_graph, 'A', 'E'))

def a_star_search(graph, start, goal, heuristic):
    visited = set()  # To keep track of visited nodes
    queue = [(heuristic[start], 0, start)]  # Initialize the queue with the starting node, its heuristic value, and cost

    while queue:
        _, cost, node = queue.pop(0)  # Dequeue the node with the lowest f(n) = g(n) + h(n)
        if node not in visited:
            print(node)  # Process the node (e.g., print it)
            visited.add(node)  # Mark the node as visited
            if node == goal:
                return node
            # Enqueue all unvisited neighbors with their cumulative costs and heuristic values
            for neighbor, edge_cost in graph[node]:
                if neighbor not in visited:
                    new_cost = cost + edge_cost
                    queue.append((new_cost + heuristic[neighbor], new_cost, neighbor))
                    queue.sort()  # Sort the queue based on f(n) values
    return None
# Example usage:
print("\nA* Search from 'A' to 'E':")
print("Result:", a_star_search(weighted_graph, 'A', 'E', heuristic))  
