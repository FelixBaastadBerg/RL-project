import random

def can_place_tree(grid, tree_shape, x, y):
    for i in range(len(tree_shape)):
        for j in range(len(tree_shape[0])):
            if tree_shape[i][j] == 1:
                if grid[y + i][x + j] != 'g':
                    return False
    return True

def place_tree(grid, tree_shape, x, y):
    for i in range(len(tree_shape)):
        for j in range(len(tree_shape[0])):
            if tree_shape[i][j] == 1:
                grid[y + i][x + j] = 'd'

def generate_environment():
    grid_size = 20
    grid = [['g' for _ in range(grid_size)] for _ in range(grid_size)]

    tree_shape = [
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]
    ]

    num_trees = 1
    tree_positions = []
    for _ in range(num_trees):
        attempts = 0
        while True:
            x = random.randint(0, grid_size - len(tree_shape[0]))
            y = random.randint(0, grid_size - len(tree_shape))
            if can_place_tree(grid, tree_shape, x, y):
                place_tree(grid, tree_shape, x, y)
                tree_positions.append((x, y))
                break
            attempts += 1
            if attempts > 1000:
                print("Failed to place a tree after 1000 attempts.")
                break

    # Write the grid to environment.txt
    with open('data/animal_map.txt', 'w') as f:
        for row in grid:
            line = ';'.join(row)
            f.write(line + '\n')

    return grid, tree_positions

def generate_monster_map():
    grid_size = 20
    grid = [[';' for _ in range(grid_size)] for _ in range(grid_size)]

    num_monsters = 0
    placed_positions = set()
    while len(placed_positions) < num_monsters:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        if (x, y) not in placed_positions:
            grid[y][x] = 'm'
            placed_positions.add((x, y))

    # Write the grid to monster_map.txt
    with open('data/monster_map.txt', 'w') as f:
        for row in grid:
            line = ''.join(row)
            f.write(line + '\n')

def generate_apple_map(environment_grid):
    grid_size = 20
    grid = [[';' for _ in range(grid_size)] for _ in range(grid_size)]

    # Collect positions where 'd' (trees) are located
    tree_positions = []
    for y in range(grid_size):
        for x in range(grid_size):
            if environment_grid[y][x] == 'd':
                tree_positions.append((x, y))

    if len(tree_positions) < 5:
        print("Not enough tree positions to place all apples.")
        return

    # Randomly select 5 positions within the trees to place apples
    apple_positions = random.sample(tree_positions, 5)
    for x, y in apple_positions:
        grid[y][x] = 'a'

    # Write the grid to apple_map.txt
    with open('data/apple_map.txt', 'w') as f:
        for row in grid:
            line = ''.join(row)
            f.write(line + '\n')

def main():
    environment_grid, tree_positions = generate_environment()
    generate_monster_map()
    generate_apple_map(environment_grid)

if __name__ == "__main__":
    main()
