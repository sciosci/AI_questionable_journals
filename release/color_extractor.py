import cv2
import matplotlib.colors as mcolors
from skimage.color import deltaE_cie76
from sklearn.cluster import MiniBatchKMeans

"""
get colors for each image
"""
def check_color(color,color_names,color_rgb):
    ind = -99
    min_diff = 999999
    for i,c in enumerate(color_rgb):
        diff = deltaE_cie76(color, c)
        if diff < min_diff:
            min_diff = diff
            ind = i
    return color_names[ind]

def primary_colors(group):
    if group == 'css':
        colors = mcolors.CSS4_COLORS
        color_names = []
        color_rgb = []
        for name, color in colors.items():
            color_names.append(name)
            color_rgb.append([int(c*255)for c in mcolors.to_rgb(color)])

    return color_names, color_rgb

def get_color(image_path, number_of_colors, color_names, color_rgb, random_seed):

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = MiniBatchKMeans(n_clusters=number_of_colors,random_state=random_seed)
    labels = clf.fit_predict(modified_image)

    center_colors = clf.cluster_centers_

    #counts = Counter(labels)
    #ordered_colors = [center_colors[i-1] for i in counts.keys()]

    return [check_color(center_colors[i],color_names,color_rgb) for i in range(number_of_colors)]


