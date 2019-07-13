import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import permutations
from fastai.vision.image import image2np
from google.colab import widgets
from fastai.vision import *
from fastai.tabular import *

class TabularAnalysis():
  def __init__(self, interp:ClassificationInterpretation, classlist:list):
    self.interp = interp
    if str(type(interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      self.means = interp.learn.data.train_ds.x.processor[0].procs[2].means
      self.stds = interp.learn.data.train_ds.x.processor[0].procs[2].stds
    self.show_losses(classlist)
    
  def create_graphs(self, df:pd.DataFrame, cat_names:list):
    cols = math.ceil(math.sqrt(len(df.columns)))
    rows = math.ceil(len(df.columns)/cols)
    df.columns = df.columns.get_level_values(0)
    tbnames = list(df.columns)
    tbnames.sort()
    tb = widgets.TabBar(tbnames)
    for i, tab in enumerate(tbnames):
      with tb.output_to(i):
        row = (int)(i / cols)
        col = i % cols
        if tab in cat_names:
          vals = pd.value_counts(df[tab])
          fig = vals.plot(kind='bar', title=tbnames[i]+' distrobution', rot=30)
        else:
          vals = df[tab].astype(float)
          vals = vals * self.stds[tab] + self.means[tab]
          sns.distplot(vals).set_title(tbnames[i]+' distrobution')
  
  def show_losses(self, classl:list, **kwargs):
    if len(classl) > 2:
      print('Warning: More than two classes has not been implemented yet.\nThe current layout is first position will be the truth, the second will be prediction')
    tl_val, tl_idx = self.interp.top_losses(len(self.interp.losses))
    classes = self.interp.data.classes
    cat_names = self.interp.data.x.cat_names
    cont_names = self.interp.data.x.cont_names
    df = pd.DataFrame(columns=[cat_names + cont_names])
    for i, idx in enumerate(tl_idx):
      da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
      cl = int(cl)
      t1 = str(da)
      t1 = t1.split(';')
      if classes[self.interp.pred_class[idx]] == classl[0] and classes[cl] == classl[1]:
        arr = []
        for x in range(len(t1)-1):
            _, value = t1[x].rsplit(' ', 1)
            arr.append(value)
        df.loc[i] = arr
    self.create_graphs(df, cat_names)
  
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
