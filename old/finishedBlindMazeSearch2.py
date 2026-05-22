# CAP 6635  Artificial Intelligence, 2024 Spring
# Jan. 22 2024, X. Zhu
# code credit: Adapted from following github project with revision
# https://gist.github.com/Nicholas-Swift/003e1932ef2804bebef2710527008f44
# List as FIFO (first in, first out)
stack = ["a", "b", "c"]

# add an element to the end of the list
stack.append("e")
stack.append("f")
print('stack after e and f added:')
print(stack)

# pop operation
d=stack.pop(0)

print(d)
print('stack after stack.pop(0) operation:')
print(stack)

# pop operation
stack.pop(0)
print('stack after another stack.pop(0) operation:')
print(stack)

# push operation
stack.append("d")
print('stack after d appended:')
print(stack)
# ['a', 'b', 'c', 'e', 'f']
# a
# ['b', 'c', 'e', 'f']
# ['c', 'e', 'f']
# ['c', 'e', 'f', 'd']
# List as LIFO (least in, first out)
stack = ["a", "b", "c"]

# add an element to the end of the list
stack.insert(0,"e")
stack.insert(0,"f")
print(stack)

# pop operation
stack.pop(0)
print(stack)

# pop operation
stack.pop(0)
print(stack)

# push operation
stack.insert(0,"d")
print(stack)
['f', 'e', 'a', 'b', 'c']
['e', 'a', 'b', 'c']
['a', 'b', 'c']
['d', 'a', 'b', 'c']
import numpy as np
import matplotlib.pyplot as plt
class Node():
    """A search node class for Maze Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.c = 0  # cost from source to current node

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):               #<-- added a hash method
        return hash(self.position)


def search_path(maze, start, end, method='DFS'):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.c = 0
    end_node = Node(None, end)
    end_node.c = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = set()                # <-- closed_list must be a set

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    expanded_nodes=0
    queue_size=0
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        #print(current_node.position[0],current_node.position[1])
        current_index = 0

        # Pop current off open list, add to closed list
        # depending on how the nodes are added to the queue, this will implement either FIFO (BFS), or LIFO (DFS)
        open_list.pop(current_index)
        closed_list.add(current_node)     # <-- change append to add

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return(expanded_nodes,queue_size,path[::-1]) # Return reversed path

        # Generate children
        expanded_nodes=expanded_nodes+1
        if(len(open_list)>queue_size):
            queue_size=len(open_list)  # check maximum queue size
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:              # <-- remove inner loop so continue takes you to the end of the outer loop
                continue

            # Create the updated cost values
            # child.c = current_node.c + 1           # diagonal and horizontal/vertical cost same
            child.c = current_node.c + np.sqrt(np.square(child.position[0] - current_node.position[0])+np.square(child.position[1] - current_node.position[1]))
            

            # Child is already in the open list
            childAlreadyExist=False
            for open_node in open_list:
                if child == open_node and child.c >= open_node.c:
                    childAlreadyExist=True
                    break

            # Add the child to the open list
            if(not childAlreadyExist):
                if method=='BFS':
                    open_list.append(child)
                if method=='DFS':
                    open_list.insert(0,child)
                #print(child.position)
            
def pathLength(path):
    dis=0
    for i in range(len(path)-1):
        x1=path[i][0]
        y1=path[i][1]
        x2=path[i+1][0]
        y2=path[i+1][1]
        dis=dis+np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return(dis)
def main():

    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    for row in maze:
        print(row)

    start = (0, 0)
    goals = [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
    
    bfs_fringe_sizes = []
    dfs_fringe_sizes = []
    goal_labels = [f"(0,{g[1]})" for g in goals]
    
    print("\n" + "="*60)
    print("RUNNING SEARCHES FOR MULTIPLE GOALS")
    print("="*60)
    
    for goal in goals:
        print(f"\nGoal: {goal}")
        
        # BFS
        expanded_nodes_bfs, queue_size_bfs, path_bfs = search_path(maze, start, goal, 'BFS')
        bfs_fringe_sizes.append(queue_size_bfs)
        print(f"  BFS - Expanded nodes: {expanded_nodes_bfs}, Max fringe size: {queue_size_bfs}")
        
        # DFS
        expanded_nodes_dfs, queue_size_dfs, path_dfs = search_path(maze, start, goal, 'DFS')
        dfs_fringe_sizes.append(queue_size_dfs)
        print(f"  DFS - Expanded nodes: {expanded_nodes_dfs}, Max fringe size: {queue_size_dfs}")
    
    # Create plots
    x_pos = np.arange(len(goals))
    
    # BFS plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_pos, bfs_fringe_sizes, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Goal State', fontsize=12)
    plt.ylabel('Maximum Fringe Size', fontsize=12)
    plt.title('BFS Maximum Fringe Size vs Goal State', fontsize=12)
    plt.xticks(x_pos, goal_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # DFS plot
    plt.subplot(1, 2, 2)
    plt.plot(x_pos, dfs_fringe_sizes, 'r-s', linewidth=2, markersize=8)
    plt.xlabel('Goal State', fontsize=12)
    plt.ylabel('Maximum Fringe Size', fontsize=12)
    plt.title('DFS Maximum Fringe Size vs Goal State', fontsize=12)
    plt.xticks(x_pos, goal_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS: Fringe Size Growth")
    print("="*60)
    print(f"\nBFS Fringe Sizes: {bfs_fringe_sizes}")
    print(f"DFS Fringe Sizes: {dfs_fringe_sizes}")
    print("\nBFS Growth Pattern: Generally increases gradually with depth")
    print("  - BFS explores level by level, so fringe holds frontier nodes at current level")
    print("  - Fringe size ≈ branching factor ^ depth")
    print(f"  - Range: {min(bfs_fringe_sizes)} to {max(bfs_fringe_sizes)}")
    print("\nDFS Growth Pattern: Highly variable, often much larger than BFS")
    print("  - DFS goes deep before backtracking, accumulating fringe nodes")
    print("  - Fringe can spike due to deep exploration paths")
    print(f"  - Range: {min(dfs_fringe_sizes)} to {max(dfs_fringe_sizes)}")
    
    return (bfs_fringe_sizes, dfs_fringe_sizes)

print(main())