import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from fastai.vision.image import image2np
from google.colab import widgets
from fastai.vision import ClassificationInterpretation

def ClassLosses(interp:ClassificationInterpretation, k:float, classes:list, **kwargs):
    if ('figsize' in kwargs):
        figsize = kwargs['figsize']
    else:
        figsize = (8,8)
    comb = list(permutations(classes, 2))
    val, idxs = interp.top_losses(len(interp.losses))

    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k/cols)

    classes_gnd = interp.data.classes

    vals = interp.most_confused()
    ranges = []
    tbnames = []

    
    for x in iter(vals):
      for y in iter(comb):
        if x[0:2] == y:
          ranges.append(x[2])
          tbnames.append(str(x[0] + ' | ' + x[1])) 
  
    tb = widgets.TabBar(tbnames)
        
    for i, tab in enumerate(tbnames):
      with tb.output_to(i):
        x = 0
        
        if ranges[i] < k:
          cols = math.ceil(math.sqrt(ranges[i]))
          rows = math.ceil(k/cols)
        if ranges[i] < 4:
          cols = 2
          rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        [axi.set_axis_off() for axi in axes.ravel()]
        for j, idx in enumerate(idxs):
          if k < x+1 or x > ranges[i]:
            break
          da, cl = interp.data.dl(interp.ds_type).dataset[idx]
          row = (int)(x / cols)
          col = x % cols

          ix = int(cl)
          if str(cl) == tab[0] and str(classes_gnd[interp.pred_class[idx]]) == tab[1]:
            da = image2np(da.data*255).astype(np.uint8)
            axes[row, col].imshow(da)
            x += 1
        plt.tight_layout()
