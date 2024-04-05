import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display


# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED     = '#FFC4CC';
LIGHT_GREEN   = '#95FD99';
BLACK         = '#000000';
GREY          = '#F0F0F0';  
WHITE         = '#FFFFFF';
GREY          = '#D1CDDC';
LIGHT_PURPLE  = '#E8D0FF';
LIGHT_ORANGE  = '#FAE0C3';
BRIGHT_PURPLE = '#8000FF';
BRIGHT_ORANGE = '#F57A09';

actions_names = {
        0: "stay",
        1: "left",
        2: "right",
        3: "up",
        4: "down"
    }


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

# def animate_solution(maze, path, policy, env, method):

# 	actions_names = {
#         0: "stay",
#         1: "left",
#         2: "right",
#         3: "up",
#         4: "down"
#     }

# 	# Map a color to each cell in the maze
# 	col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: BRIGHT_PURPLE, 4: GREY, 5: BRIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};
# 	# Size of the maze
# 	rows,cols = maze.shape;

	

# 	if method == 'ValIter-Key':
# 		new_path = [env.new_num_states_to_position[s] for s in path]
# 		path_agent = [pos[0] for pos in new_path]
# 		path_minotaur = [pos[1] for pos in new_path]
# 		path_key = [pos[2] for pos in new_path]

# 	else:
# 		new_path = [env.num_states_to_position[s] for s in path]
# 		path_agent = [pos[0] for pos in new_path]
# 		path_minotaur = [pos[1] for pos in new_path]

# 	# Create figure of the size of the maze
# 	fig = plt.figure(1, figsize=(cols,rows))
# 	# Remove the axis ticks and add title title
# 	ax = plt.gca()
# 	ax.set_title('Policy simulation')
# 	ax.set_xticks([])
# 	ax.set_yticks([])

# 	# Give a color to each cell
# 	colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
# 	# Create figure of the size of the maze
# 	fig = plt.figure(1, figsize=(cols,rows))

# 	# Create a table to color
# 	grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

# 	# Modify the hight and width of the cells in the table
# 	tc = grid.properties()['children']
# 	for cell in tc:
# 		cell.set_height(1.0/rows)
# 		cell.set_width(1.0/cols)

# 	# Update the color at each frame
# 	for i in range(len(path_agent)):
# 		if method == 'ValIter-Key':
# 			_, pos, key = env.new_num_states_to_position[path[i]]

# 		else:
# 			_, pos = env.num_states_to_position[path[i]]


# 		for ii in range(maze.shape[0]):
# 			for jj in range(maze.shape[1]):
# 				if maze[(ii,jj)] != 1:
# 					if method == 'ValIter':
# 						s = env.position_states_to_num[((ii,jj), pos)]
# 						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s]])

# 					elif method == 'DynProg':
# 						s = env.position_states_to_num[((ii,jj), pos)]
# 						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s,i]])

# 					elif method == 'ValIter-Key':
# 						s = env.new_position_states_to_num[((ii,jj), pos, key)]
# 						grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s]])
# 					#draw_arrow(ax, (jj, ii), policy[s])

# 		grid.get_celld()[(path_agent[i])].set_facecolor(BRIGHT_ORANGE)
# 		grid.get_celld()[(path_agent[i])].get_text().set_text('Player')
# 		# MINOTAUR FRAME 
# 		grid.get_celld()[(path_minotaur[i])].set_facecolor(BRIGHT_PURPLE)
# 		grid.get_celld()[(path_minotaur[i])].get_text().set_text('Minotaur')
# 		# EXIT FRAME
# 		grid.get_celld()[(6,5)].set_facecolor(LIGHT_GREEN)
# 		grid.get_celld()[(6,5)].get_text().set_text('Exit')

# 		display.display(fig)
# 		display.clear_output(wait=True)
# 		time.sleep(1)

# 		grid.get_celld()[(path_agent[i])].set_facecolor(col_map[maze[path_agent[i]]])
# 		grid.get_celld()[(path_agent[i])].get_text().set_text('')
# 		grid.get_celld()[(path_minotaur[i])].set_facecolor(col_map[maze[path_minotaur[i]]])
# 		grid.get_celld()[(path_minotaur[i])].get_text().set_text('')
		
# 		ax.clear()
# 		ax.set_title('Policy simulation')
# 		ax.set_xticks([])
# 		ax.set_yticks([])

# 		# Give a color to each cell
# 		colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
# 		# Create a table to color
# 		grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

# 		# Modify the hight and width of the cells in the table
# 		tc = grid.properties()['children']
# 		for cell in tc:
# 			cell.set_height(1.0/rows)
# 			cell.set_width(1.0/cols)
			

# def draw_arrow(ax, cell, direction, color='black'):
#     y, x = cell
#     dx, dy = 0, 0
#     if direction == 3:
#         dy = -0.25
#     elif direction == 4:
#         dy = 0.25
#     elif direction == 2:
#         dx = 0.25
#     elif direction == 1:
#         dx = -0.25
#     # elif direction == 0:
# 	# 	pass
#     # Option to indicate a stop (for example, a point or a small star)
#     ax.scatter(x+0.25, y-0.25, color=color, marker='*')
#     return ax.arrow(x+0.25, y+0.25, dx, dy, head_width=0.2, head_length=0.2, fc=color, ec=color)




def animate_solution(env, path, policy):
	# Map a color to each cell in the maze
	col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};
	# Size of the maze
	maze = env.maze
	rows,cols = maze.shape;
	
	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))
	# Remove the axis ticks and add title title
	ax = plt.gca()
	ax.set_title('Policy simulation')
	ax.set_xticks([])
	ax.set_yticks([])
	# Give a color to each cell
	colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))
	# Create a table to color
	grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

	# Modify the hight and width of the cells in the table
	tc = grid.properties()['children']
	for cell in tc:
		cell.set_height(1.0/rows)
		cell.set_width(1.0/cols)

	# Update the color at each frame
	for i in range(len(path)):
		pos = env.num_states_to_position[path[i]]
		for ii in range(maze.shape[0]):
			for jj in range(maze.shape[1]):
				if maze[(ii,jj)] != 1:
					s = env.position_states_to_num[(ii,jj)]
					grid.get_celld()[(ii,jj)].get_text().set_text(actions_names[policy[s,i]])

		grid.get_celld()[pos].set_facecolor(BRIGHT_ORANGE)
		grid.get_celld()[pos].get_text().set_text('Player')
		# EXIT FRAME
		grid.get_celld()[env.exit].set_facecolor(LIGHT_GREEN)
		grid.get_celld()[env.exit].get_text().set_text('Exit')

		display.display(fig)
		display.clear_output(wait=True)
		time.sleep(1)
		grid.get_celld()[pos].set_facecolor(col_map[maze[pos]])
		grid.get_celld()[pos].get_text().set_text('')
	
		ax.clear()
		ax.set_title('Policy simulation')
		ax.set_xticks([])
		ax.set_yticks([])

		# Give a color to each cell
		colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
		# Create a table to color
		grid = plt.table(cellText=None,cellColours=colored_maze,cellLoc='center',loc=(0,0),edges='closed')

		# Modify the hight and width of the cells in the table
		tc = grid.properties()['children']
		for cell in tc:
			cell.set_height(1.0/rows)
			cell.set_width(1.0/cols)