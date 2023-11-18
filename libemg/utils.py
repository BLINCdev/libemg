import os

import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def get_windows(data, window_size, window_increment):
    """Extracts windows from a given set of data.

    Parameters
    ----------
    data: list
        An NxM stream of data with N samples and M channels
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.

    Returns
    ----------
    list
        The set of windows extracted from the data as a NxCxL where N is the number of windows, C is the number of channels 
        and L is the length of each window. 

    Examples
    ---------
    >>> data = np.loadtxt('data.csv', delimiter=',')
    >>> windows = get_windows(data, 100, 50)
    """
    num_windows = int((data.shape[0]-window_size)/window_increment) + 1
    windows = []
    st_id=0
    ed_id=st_id+window_size
    for w in range(num_windows):
        windows.append(data[st_id:ed_id,:].transpose())
        st_id += window_increment
        ed_id += window_increment
    return np.array(windows)

def _get_mode_windows(data, window_size, window_increment):
    windows = get_windows(data, window_size, window_increment)
    # we want to get the mode along the final dimension
    mode_of_windows = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=windows.astype(np.int64))
    
    return mode_of_windows.squeeze()

def _get_fn_windows(data, window_size, window_increment, fn):
    windows = get_windows(data, window_size, window_increment)
    # we want to apply the function along the final dimension
    
    if type(fn) is list:
        fn_of_windows = windows
        for i in fn:
            fn_of_windows = np.apply_along_axis(lambda x: i(x), axis=2, arr=fn_of_windows)
    else:
        fn_of_windows = np.apply_along_axis(lambda x: fn(x), axis=2, arr=windows)
    return fn_of_windows.squeeze()

def make_regex(left_bound, right_bound, values=[]):
    """Regex creation helper for the data handler.

    The OfflineDataHandler relies on regexes to parse the file/folder structures and extract data. 
    This function makes the creation of regexes easier.

    Parameters
    ----------
    left_bound: string
        The left bound of the regex.
    right_bound: string
        The right bound of the regex.
    values: list
        The values between the two regexes.

    Returns
    ----------
    string
        The created regex.
    
    Examples
    ---------
    >>> make_regex(left_bound = "_C_", right_bound="_EMG.csv", values = [0,1,2,3,4,5])
    """
    left_bound_str = "(?<="+ left_bound +")"
    mid_str = "(?:"
    for i in values:
        mid_str += i + "|"
    mid_str = mid_str[:-1]
    mid_str += ")"
    right_bound_str = "(?=" + right_bound +")"
    return left_bound_str + mid_str + right_bound_str

def make_gif(frames, output_filepath = 'libemg.gif', duration = 100):
    """Save a .gif video file from a list of images.


    Parameters
    ----------
    frames: list
        List of frames, where each element is a PIL.Image object.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    
    """
    frames[0].save(
        output_filepath,
        save_all=True,
        append_images=frames[1:],   # append remaining frames
        format='GIF',
        duration=duration,
        loop=0  # infinite loop
    )

def make_gif_from_directory(directory_path, output_filepath = 'libemg.gif', match_filename_function = None, 
                            delete_images = False, duration = 100):
    """Save a .gif video file from image files in a specified directory. Accepts all image types that can be read using
    PIL.Image.open(). Appends images in alphabetical order.


    Parameters
    ----------
    directory_path: string
        Path to directory that contains images.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    match_filename_function: Callable or None (optional), default=None
        Match function that determines which images in directory to use to create .gif. The match function should only expect a filename
        as a parameter and return True if the image should be used to create the .gif, otherwise it should return False. 
        If None, reads in all images in the directory.
    delete_images: bool (optional), default=False
        True if images used to create .gif should be deleted, otherwise False.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    """
    if match_filename_function is None:
        # Combine all images in directory
        match_filename_function = lambda x: True
    frames = []
    filenames = os.listdir(directory_path)
    filenames.sort()    # sort alphabetically
    matching_filenames = [] # images used to create .gif

    for filename in filenames:
        absolute_path = os.path.join(directory_path, filename)
        if match_filename_function(filename):
            # File matches the user pattern and is an accepted image format
            try:
                image = Image.open(absolute_path)
                frames.append(image)
                matching_filenames.append(absolute_path)
            except UnidentifiedImageError:
                # Reading non-image file
                print(f'Skipping {absolute_path} because it is not an image file.')
    
    # Make .gif from frames
    make_gif(frames, output_filepath, duration=duration)

    if delete_images:
        # Delete all images used to create .gif
        for filename in matching_filenames:
            os.remove(filename)
            
def _convert_plot_to_image(fig):
    """Convert a matplotlib Figure to a PIL.Image object.

    Parameters
    ----------
    fig: matplotlib.pyplot.Figure
        Figure that should be saved as an image.
    """
    canvas = FigureCanvasAgg(fig)
    
    # Get RGBA buffer
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()

    # Convert the buffer to a PIL Image
    return Image.frombytes('RGBA', canvas.get_width_height(), rgba_buffer)

def _add_image_label_axes(fig):
    """Add axes to a matplotlib Figure for displaying images in the top, right, bottom, and left of the Figure. 
    
    Parameters
    ----------
    fig: matplotlib.pyplot.Figure
        Figure to add image axes to.
    """
    # Make 3 x 3 grid
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

    # Create subplots using the gridspec
    ax_main = plt.subplot(gs[1, 1])  # Main plot
    ax_right = plt.subplot(gs[1, 2])  # Right side subplot
    ax_left = plt.subplot(gs[1, 0])   # Left side subplot
    ax_top = plt.subplot(gs[0, 1])    # Top subplot
    ax_bottom = plt.subplot(gs[2, 1]) # Bottom subplot
    
    image_axs = (ax_top, ax_right, ax_bottom, ax_left)
    
    # Turn off axes for image axes
    for ax in image_axs:
        ax.axis('off')
    return (ax_main, ax_top, ax_right, ax_bottom, ax_left)

def make_regression_training_gif(coordinates, output_filepath = 'libemg.gif', duration = 100, xlabel = '', ylabel = '', axis_images = None, save_coordinates = False):
    """Save a .gif file of an icon moving around a 2D plane. Can be used for regression training.
    
    Parameters
    ----------
    coordinates: numpy.ndarray
        N x M matrix, where N is the number of frames and M is the number of DOFs. Order is x-axis, y-axis, and angle (degrees counter-clockwise).
        Each row contains the value for x position, y position, and / or angle depending on how many DOFs are passed in.
    output_filepath: string (optional), default='libemg.gif'
        Filepath of output file.
    duration: int (optional), default=100
        Duration of each frame in milliseconds.
    xlabel: string (optional), default=''
        Label for x-axis.
    ylabel: string (optional), default=''
        Label for y-axis.
    axis_images: dict (optional), default=None
        Dictionary mapping compass directions to images. Images will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
        Valid keys are 'N', 'E', 'S', and 'W'.
    save_coordinates: bool (optional), default=False
        True if coordinates should be saved to a .txt file for ground truth values, otherwise False.
    """
    # Plotting functions
    def plot_dot(frame_coordinates):
        # Parse coordinates
        x = frame_coordinates[0]
        y = frame_coordinates[1]
        # Dot properties
        size = 50
        colour = 'black'
        plt.scatter(x, y, s=size, c=colour)
    
    def plot_arrow(frame_coordinates):
        # Parse coordinates
        x_tail = frame_coordinates[0]
        y_tail = frame_coordinates[1]
        angle_degrees = frame_coordinates[2]
        # Convert angle to radians
        arrow_angle_radians = np.radians(angle_degrees)
        # Arrow properties
        arrow_length = 0.1
        head_size = 0.05
        arrow_colour = 'black'
        # Calculate arrow head coordinates
        x_head = x_tail + arrow_length * np.cos(arrow_angle_radians)
        y_head = y_tail + arrow_length * np.sin(arrow_angle_radians)
        plt.arrow(x_tail, y_tail, x_head - x_tail, y_head - y_tail, head_width=head_size, head_length=head_size, fc=arrow_colour, ec=arrow_colour)
    
    # Plot a dot if 2 DOFs were passed in, otherwise plot arrow
    axis_limits = (-1.2, 1.2)
    plot_icon = plot_dot if coordinates.shape[1] == 2 else plot_arrow
    frames = []
    for frame_coordinates in coordinates:
        fig = plt.figure()
        
        if axis_images is not None:
            ax_main, ax_top, ax_right, ax_bottom, ax_left = _add_image_label_axes(fig)
            loc_axis_map = {
                'N': ax_top,
                'E': ax_right,
                'S': ax_bottom,
                'W': ax_left
            }
            for loc, image in axis_images.items():
                ax = loc_axis_map[loc]
                ax.imshow(image)
            plt.sca(ax_main)    # set main axis so icon is drawn correctly
        
        # Plot icon
        plot_icon(frame_coordinates)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(axis_limits[0], axis_limits[1]) # restrict axis to -1, 1 for visual clarity and proper icon size
        plt.ylim(axis_limits[0], axis_limits[1]) # restrict axis to -1, 1 for visual clarity and proper icon size
        plt.tight_layout()
        frame = _convert_plot_to_image(fig)
        frames.append(frame)
        plt.close() # close figure
    
    # Save file
    make_gif(frames, output_filepath=output_filepath, duration=duration)
    if save_coordinates:
        # Save coordinates in .txt file
        filename_no_extension = os.path.splitext(output_filepath)[0]
        labels_filepath = filename_no_extension + '.txt'
        np.savetxt(labels_filepath, coordinates, delimiter=',')
