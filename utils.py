'''
=======================================================================================
DSProjectUtils.py

Nadav Kahlon & Amir Deitch, June 2022.
=======================================================================================

This file represents a utility module for our Data Science project notebook.
It contains some "thick" utility methods regarding shallow matters such as formatting
  print output while discussing less important issues in the project.
We decided to write those utility functions here (separately) in order to make our
  discussions look more clean. Of course, we don't use those functions in more central
  parts of the project.

'''

import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import math
import numpy as np
from xml.dom import minidom
import pandas as pd


'''
Applies color to a string representing an XML component.
Input:
  > 's' - the input string.
  > 'component' - a string representing the XML component; should be one of the 
    following: "tag", "attrib", "attrib_val"
Returns the colored string.
'''
def color_xml(s, component):
    # pick a style format based on the componenet
    if component is "tag": color = '31' # red
    elif component is "attrib": color = '35' # purple
    elif component is "attrib_val": color = '32' # green
    else: color = ''

    # apply it
    return '\x1b[%sm%s\x1b[0m' % (color, s)

'''
Creates a one-line string representation of an xml.etree.ElementTree.Element.
Input:
  > 'node' - the ElementTree node in question.
Returns a string of the format:
  '{tag}[ {attribute}={attribute_val}]*'
'''
def format_etree_node(node):
    # string to hold the result
    result = color_xml(node.tag, "tag")

    # append information about the node's attributes
    for attrib in node.attrib.keys():
        attrib_val = f'"{node.attrib[attrib]}"'
        result += f' {color_xml(attrib,"attrib")}={color_xml(attrib_val, "attrib_val")}'

    return result

'''
Returns a multiple-line recursive string representation of an xml.etree.ElementTree.
Input:
  > 'root' - an xml.etree.ElementTree.Element representing the root of the tree.
  > 'max_seq_len' - the maximum number of child nodes of the same tag appearing in
    sequence that will be fully-printed. Beyond that, a line saying that there are
    more nodes of that tag will be added. We assume it's positive. Default: 3.
  > 'prefix' - some prefix to add to the beginning of each line of the string (e.g. a
    tab). Default: ''.
'''
def format_etree(root, max_seq_len=3, prefix=''):
    if len(root) > 0:
        # if the root has children - format it and include its children
        result = prefix + f'<{format_etree_node(root)}>\n' +\
                 format_etree_descendants(root, max_seq_len, prefix+'\t') + \
                 prefix + f'</{color_xml(root.tag,"tag")}>'
        return result
    
    else:
        # otherwise - it should be a simple one-line representation
        return prefix + f'<{format_etree_node(root)} />'

'''
Returns a recursive multiple-line string representation of the descendants of an
  xml.etree.ElementTree, truncating a long sequence of children with the same tag
  name.
Input:
  > 'node' - the node whose descendants we care about.
  > 'max_seq_len' - the maximum number of nodes of the same tag appearing in
    sequence that will be fully-printed. Beyond that, a line saying that there are
    more nodes of that tag will be added. We assume it's positive. Default: 3.
  > 'prefix' - some prefix to add to the beginning of each line of the string (e.g. a
    tab). Default: ''.
'''
def format_etree_descendants(node, max_seq_len=3, prefix=''):
    # variable to hold the resulting string
    result = ''

    # variables to keep track of the last sequence of children of the same tag
    seq_len = 0
    last_child = None

    # scan the children
    for child in node:

        if last_child == None or last_child.tag != child.tag:
            # if we should start a new sequence
            if seq_len > max_seq_len:
                # if we should finish off another long sequence
                result += prefix + f'[... {seq_len-max_seq_len} more ' \
                                + f'{color_xml(last_child.tag,"tag")} nodes]\n'
            seq_len = 0
        
        # if we're not in an ongoing long sequence - add information about this child
        if seq_len < max_seq_len:
            result += format_etree(child, max_seq_len, prefix) + '\n'

        # update sequence info
        seq_len += 1
        last_child = child
    
    # finish off one last long sequence
    if seq_len > max_seq_len:
        result += prefix + f'[... {seq_len-max_seq_len} more ' \
                         + f'{color_xml(last_child.tag,"tag")} nodes]\n'

    return result

'''
Plot a tile extracted from a whole-slide-image containing cellmarks.
Input:
  > 'ax' - the matplotlib axis object to plot on.
  > 'tile_img' - the extracted tile image.
  > 'tile_cellmarks' - list of cellmarks represented by pairs ((x1,y1),(x2,y2))
    of vertix coordinates.
'''
def plot_tile(ax, tile_img, tile_cellmarks):
    # plot tile
    ax.imshow(tile_img)
    # plot the cellmarks it contains
    for ((x1,y1), (x2,y2)) in tile_cellmarks:
        ax.plot((x1,x2), (y1,y2), linewidth=2.5, color='lime')
    # organize plot
    ax.axis('off')
    cellmark_line = mlines.Line2D([], [], color='lime', label='Cellmarks', linewidth=2.5)
    ax.legend(handles=[cellmark_line])

    
'''
Plot a patch of a whole-slide-image with green circles representing pinned 1-point
  markers.
Input:
  > 'ax' - the matplotlib axis object to plot on.
  > 'patch' - the extracted patch.
  > 'pins' - list of pairs (y,x) representing the pinned 1-point markers (y and x
    should both be one-value PyTorch tensors).
'''
def plot_pinned_patch(ax, patch, pins):
    # plot patch
    ax.imshow(patch)
    # plot the cellmarks it contains
    for (y,x) in pins:
        x = x.cpu(); y = y.cpu()
        ax.add_patch(plt.Circle((x,y),color='g', alpha=0.25, radius=30))

'''
Plots a set of feature maps.
Input:
  > 'fmap' - tensor of shape (N[, {1,3}], h, w) whose entries are the feature maps
    you wish to display.
  > 'layout_ratio - a pair (H,W) of floating point value, representing the ratio
    between the number of rows of subplots and the number of column of subplots.
    Deafult: None, which means that an horizontal sequence of subplots will be displayed.
  > 'cmap' - colormap for plots. Default: None (default colormap).
  > 'to_normalize' - a boolean stating whether each feature map should be normalized
    to a range between 0.0 and 1.0. Default: True.
'''
def plot_fmap(fmap, layout_ratio=None, cmap=None, to_normalize=True):
    # calculate the shape of the layout of the feature maps
    N = fmap.shape[0]
    if layout_ratio is not None:
        hratio = layout_ratio[0] / layout_ratio[1]
        hside = math.ceil(math.sqrt(hratio * N))
        wratio = layout_ratio[1] / layout_ratio[0]
        wside = math.ceil(math.sqrt(wratio * N))
        layout= (hside, wside)
    else:
        layout = (1,N)


    # organize map for displaying
    fmap = fmap.clone().detach().cpu()
    if len(fmap.shape) == 4:
        fmap = fmap.moveaxis(1, -1)
    if to_normalize:
        fmap = fmap.moveaxis(0,-1)
        fmap -= fmap.reshape(-1,N).min(dim=0).values
        fmap /= (fmap.reshape(-1,N).max(dim=0).values + 1e-10)
        fmap = fmap.moveaxis(-1,0)
    
    # generate subplots
    for n in range(N):
        ax = plt.subplot(*layout, n+1)
        ax.axis('off')
        ax.imshow(fmap[n], cmap=cmap)

'''
Animates the progress of an adversarial attack on a model that produces 1-channel
  maps.
Input:
  > 'x_tag_hist' - history of synthesized inputs (starting with the original).
  > 'y_pred_hist' - history of model predictions on synthesized inputs (starting
    with the original).
  > 'y_target' - target model output.
All parameters can have a singleton batch dimension.
Returns the required matplotlib animation object.
'''
def animate_attack(x_tag_hist, y_pred_hist, y_target):
    # prepare figure
    fig, axes = plt.subplots(1,3,figsize=(12,5))
    for ax in axes: ax.axis('off')

    # prepare plots
    x_tag_img = axes[0].imshow(x_tag_hist[0].squeeze().moveaxis(0,-1).cpu())
    axes[0].set_title('Input Patch')
    y_pred_img = axes[1].imshow(y_pred_hist[0].squeeze().cpu(), vmin=0, vmax=1)
    axes[1].set_title('Predicted Output')
    axes[2].imshow(y_target.squeeze().cpu(), vmin=0, vmax=1)
    axes[2].set_title('Target Output')
    
    # add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(y_pred_img, cax=cbar_ax)
    
    # prepare animation
    def drawframe(i):
        x_tag_img.set_data(x_tag_hist[i].squeeze().moveaxis(0,-1).cpu())
        y_pred_img.set_data(y_pred_hist[i].squeeze().cpu())
        return [x_tag_img, y_pred_img]
    anim = animation.FuncAnimation(
      fig, drawframe, frames=len(x_tag_hist), interval=20, blit=True
    )

    return anim

'''
Draws a matrix of predicted cells in a whole-slide image.
Input:
  > 'img_handle' - opened pytiff handle object for the whole-slide image.
  > 'cell_centers' - numpy array of shape (num_of_cells,2) whose rows are the
    center coordinates of the cells.
  > 'cell_diameters' - numpy array of shape (num_of_cells,2,2) whose (n,i)'th entry
    is the coordinate vector of the i'th edge of the n'th cell.
  > 'to_plot' - iterator of indices of the cells to include in the plot.
  > 'window_shape' - shape of window around each cell to plot. Default: (110,110).
'''
def plot_predictions(img_handle, cell_centers, cell_diameters, to_plot,
                     window_shape=(110,110)):
    # calculate side of the plots matrix
    side = math.ceil(math.sqrt(len(to_plot)))
    
    # extract window shape
    window_h, window_w = window_shape

    # process each cell
    for i, cell_idx in enumerate(to_plot):
        # extract relevant info
        r, c = cell_centers[cell_idx]
        diameter = np.array(cell_diameters[cell_idx])

        # extract patch and shift diameter
        patch = img_handle[r - window_h//2 : r + window_h//2,
                           c - window_w//2 : c + window_w//2]
        diameter -= [r - window_h//2, c - window_w//2]

        # organize subplot
        ax = plt.subplot(side, side, i+1)
        ax.set_title(f'Cell {cell_idx}')
        ax.axis('off')

        # plot patch and diameter
        ax.imshow(patch)
        ax.plot(diameter[:,0], diameter[:,1], linewidth=3, color='orange', alpha=0.7)
        ax.scatter(diameter[:,0], diameter[:,1], color='yellow', s=8)




  
'''
This function creates a pandas dataframe from an XML file of a WSI.
It does it as made inside the EDA section.
'''
def load_xml(path):
    # load xml file and fetch lines (=cells) records
    test_file = minidom.parse(path)
    test_regions = test_file.getElementsByTagName('Region')
    test_lines = [test_region
            for test_region in test_regions
            if test_region.attributes['Type'].value is '4']


    # returns [X0,Y0,X1,Y1] diameter points of line number i
    def get_test_labels(i):
        line = test_lines[i]
        # access relevant attributes through xml tree
        vertices = line.childNodes[3].childNodes

        X0 = int(float(vertices[1].attributes['X'].value))
        Y0 = int(float(vertices[1].attributes['Y'].value))
        X1 = int(float(vertices[3].attributes['X'].value))
        Y1 = int(float(vertices[3].attributes['Y'].value))

        return [X0, Y0, X1, Y1]

    # returns [attr_names],[attributes] of line i, where attributes is all line data except its diameters points
    def get_test_aux_data(i):
        line = test_lines[i]
        pairs_name_val=line.attributes.items()
        attr_names = [attr[0] for attr in pairs_name_val]
        attr_val = [attr[1] for attr in pairs_name_val]
        return attr_names,attr_val

    # returns whole data of a line as [attributes],[values]
    def get_test_line(i):
        labels=get_test_labels(i)
        attr_names,attr_vals=get_test_aux_data(i)
        values=attr_vals+labels
        names=attr_names+['X0','Y0','X1','Y1']
        return names,values

    test_all_data=[]
    test_ln_names=None
    for i in range(len(test_lines)):
        test_ln_names,ln_vals=get_test_line(i)
        test_all_data+=[ln_vals]
    test_frame=pd.DataFrame(test_all_data,columns=test_ln_names)

    # change object to float

    for col in ['Id','Length','LengthMicrons']:
        test_frame[col]=pd.to_numeric(test_frame[col])
    return test_frame
