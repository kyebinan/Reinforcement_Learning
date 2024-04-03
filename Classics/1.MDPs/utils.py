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


def draw_grid(grid, mdp):

    # Map a color to each cell in the grid
    if mdp == "Vaccum":
        col_map = {0: WHITE, 1: GREY, 2: BLACK,};
    elif mdp == "Maze":
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = grid.shape;
    colored_grid = [[col_map[grid[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the grid
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The grid');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = grid.shape;
    colored_grid = [[col_map[grid[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the grid
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_grid,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);