import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import permutations
from google.colab import widgets
from fastai.vision import ClassificationInterpretation

class ClassLosses():
  "Plot the most confused datapoints and statistics for your misses. \nPass in a `Learner` object and a list of classes to look at."
  def __init__(self, interp:ClassificationInterpretation, classlist:list):
    self.interp = interp
    if str(type(interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      self.means = interp.learn.data.train_ds.x.processor[0].procs[2].means
      self.stds = interp.learn.data.train_ds.x.processor[0].procs[2].stds
    self.show_losses(classlist)
    
  def create_graphs(self, df_list:list, cat_names:list):
    cols = math.ceil(math.sqrt(len(df_list)))
    rows = math.ceil(len(df_list)/cols)
    df_list[0].columns = df_list[0].columns.get_level_values(0)
    tbnames = list(df_list[0].columns)
    tbnames.sort()
    tbnames = tbnames[1:]
    tb = widgets.TabBar(tbnames)
    
    
    for i, tab in enumerate(tbnames):
      with tb.output_to(i):
        
        fig, ax = plt.subplots(len(df_list), figsize=(8,8))
        for j, x in enumerate(df_list):
          row = (int)(j / cols)
          col = j % cols
          if tab in cat_names:
            vals = pd.value_counts(df_list[j][tab].values.flatten())
            ttl = str.join('', df_list[j].columns[0])
            fig = vals.plot(kind='bar', title= ttl + ' ' + tbnames[i]+' distrobution', rot=30, ax=ax[j])
          else:
            vals = df_list[j][tab].astype(float)
            vals = vals * self.stds[tab] + self.means[tab]
            ttl = str.join('', df_list[j].columns[0])
            
            axs = vals.plot(kind='hist', ax=ax[j])
            vals.plot(kind='kde', ax=axs, secondary_y=True)
            
        plt.tight_layout()
  
  def show_losses(self, classl:list, **kwargs):
    if str(type(self.interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      
      tl_val, tl_idx = self.interp.top_losses(len(self.interp.losses))
      classes = self.interp.data.classes
      cat_names = self.interp.data.x.cat_names
      cont_names = self.interp.data.x.cont_names
      arr1 = []
      comb = list(permutations(classl,2))
      print('Variable Distrobution:')
      for j, x in enumerate(comb):
        df = pd.DataFrame(columns=[[str(comb[j])] + cat_names + cont_names])
        for i, idx in enumerate(tl_idx):
          da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
          cl = int(cl)
          t1 = str(da)
          t1 = t1.split(';')
          if classes[self.interp.pred_class[idx]] == comb[j][0] and classes[cl] == comb[j][1]:
            arr = []
            arr.append(str(comb[j]))
            for x in range(len(t1)-1):
                _, value = t1[x].rsplit(' ', 1)
                arr.append(value)
            df.loc[i] = arr
        arr1.append(df)
      self.create_graphs(arr1, cat_names)
      
    else:
        comb = list(permutations(classl, 2))
        tl_val, tl_idx = self.interp.top_losses(len(self.interp.losses))

        classes_gnd = self.interp.data.classes
        vals = self.interp.most_confused()
        ranges = []
        tbnames = []

        k = input('Please enter a value for `k`: ')
        k = int(k)

        for x in iter(vals):
          for y in iter(comb):
            if x[0:2] == y:
              ranges.append(x[2])
              tbnames.append(str(x[0] + ' | ' + x[1]))
        print('Misclassified Pictures:')
        tb = widgets.TabBar(tbnames)

        for i, tab in enumerate(tbnames):
          with tb.output_to(i):

            x = 0          
            if ranges[i] < k:
              cols = math.ceil(math.sqrt(ranges[i]))
              rows = math.ceil(ranges[i]/cols)

            if ranges[i] < 4:
              cols, rows = 2, 2

            else:
              cols = math.ceil(math.sqrt(k))
              rows = math.ceil(k/cols)

            fig, axes = plt.subplots(rows, cols, figsize=(8,8))
            [axi.set_axis_off() for axi in axes.ravel()]
            for j, idx in enumerate(tl_idx):
              if k < x+1 or x > ranges[i]:
                break
              da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
              row = (int)(x / cols)
              col = x % cols

              ix = int(cl)
              if str(cl) == tab.split(' ')[0] and str(classes_gnd[self.interp.pred_class[idx]]) == tab.split(' ')[2]:
                da = image2np(da.data*255).astype(np.uint8)
                axes[row, col].imshow(da)
                x += 1
            plt.tight_layout()

    
