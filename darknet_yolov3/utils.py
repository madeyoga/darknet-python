import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_bboxes(img, predictions):
    plt.imshow(img)
    ax = plt.gca()

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    for p in predictions:
        label, conf, (x,y,w,h) = p
        rect = Rectangle((x,y),
                         w,
                         h,
                         ec='white',
                         fc='white',
                         alpha=0.5)
        ax.add_patch(rect)

        ax.text(x,
                y-12,
                '{}: {}%'.format(label, int(conf*100)),
                fontsize=9,
                bbox={'facecolor':'white','alpha':0.5,'edgecolor':'none','pad':1},
                va='center')

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    return plt.show()
        
def decode_output(predictions):
    res = {}

    for p in predictions:
        label, _, box = p
        if not label in res:
            res[label] = []
            res[label].append(box)
        else:
            res[label].append(box)

    return res
