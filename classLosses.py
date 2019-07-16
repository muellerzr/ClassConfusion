import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import permutations
from google.colab import widgets
from fastai.vision import ClassificationInterpretation

class ClassLosses():
  """Plot the most confused datapoints and statistics for your misses. 
  \nPass in a `interp` object and a list of classes to look at. 
  Optionally you can include an odered list in the form of [[class_1, class_2]],
  \n a figure size, and a cut_off limit for the maximum categorical categories to use on a variable"""
  def __init__(self, interp:ClassificationInterpretation, classlist:list, 
               is_ordered:bool=False, cut_off:int=100, varlist:list=list(),
               figsize:tuple=(8,8)):
    self.interp = interp
    if str(type(interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      if interp.learn.data.train_ds.x.cont_names != []: 
        for x in range(len(interp.learn.data.procs)):
          if "Normalize" in str(interp.learn.data.procs[x]):
            self.means = interp.learn.data.train_ds.x.processor[0].procs[x].means
            self.stds = interp.learn.data.train_ds.x.processor[0].procs[x].stds
    self.is_ordered = is_ordered
    self.cut_off = cut_off
    self.figsize = figsize
    self.vars = varlist
    self.show_losses(classlist)
    
    
  def create_tabs(self, df_list:list, cat_names:list):
    print('Variable Distrobution:')
    self.cols = math.ceil(math.sqrt(len(df_list)))
    self.rows = math.ceil(len(df_list)/self.cols)
    if len(df_list) < 4:
      self.is_less = True
    df_list[0].columns = df_list[0].columns.get_level_values(0)
    if len(self.vars) > 0:
      self.tbnames = self.vars
    else:
      tbnames = list(df_list[0].columns)
      self.tbnames = tbnames[:-1]
    tb = widgets.TabBar(self.tbnames)
      
    for i, tab in enumerate(self.tbnames):
      with tb.output_to(i):
        self.create_graphs(tab, df_list, cat_names)

    
    
  def create_graphs(self, tab:str, df_list:list, cat_names:list):
      
      if self.is_less == True:
        fig, ax = plt.subplots(len(df_list), figsize=self.figsize)
      else:
        fig, ax = plt.subplots(self.rows, self.cols, figsize=self.figsize)
      for j, x in enumerate(df_list):
        row = (int)(j / self.cols)
        col = j % self.cols
        if tab in cat_names:
          if df_list[j][tab].nunique() > self.cut_off:
            print(f'{tab} has too many unique values to graph well.')
          else:
            vals = pd.value_counts(df_list[j][tab].values)
            ttl = str.join('', df_list[j].columns[-1])
            if j == 0:
              title = ttl + ' ' + tab +' distrobution'
            else:
              title = 'Misclassified ' + ttl + ' ' + tab+' distrobution'
            if self.is_less == True:
              fig = vals.plot(kind='bar', title= title, rot=30, ax=ax[j])
            else:
              fig = vals.plot(kind='bar', title= title, rot=30, ax=ax[row, col])
        else:
          vals = df_list[j][tab]
          ttl = str.join('', df_list[j].columns[-1])
          if j == 0:
            title = ttl + ' ' + tab + ' distrobution'
          else:
            title = 'Misclassified ' + ttl + ' ' + tab+' distrobution'
          if self.is_less == True:
            axs = vals.plot(kind='hist', ax=ax[j], title=title, y='Frequency')
          else:
            axs = vals.plot(kind='hist', ax=ax[row, col], title=title, y='Frequency')
          axs.set_ylabel('Frequency')
          if len(set(vals)) > 1:
            vals.plot(kind='kde', ax=axs, title = title, secondary_y=True)
          else:
            print('Less than two unique values, cannot graph the KDE')

      plt.tight_layout()
      
  
  def show_losses(self, classl:list, **kwargs):
    if str(type(self.interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      self.tab_losses(classl)
    else:
      self.im_losses(classl)

      
  def tab_losses(self, classl:list, **kwargs):
      tl_val, tl_idx = self.interp.top_losses(len(self.interp.losses))
      classes = self.interp.data.classes
      cat_names = self.interp.data.x.cat_names
      cont_names = self.interp.data.x.cont_names
      if self.is_ordered == False:
        comb = list(permutations(classl,2))
      else:
        comb = classl
      
      dfarr = []
      
      arr = []
      for i, idx in enumerate(tl_idx):
        da, _ = self.interp.data.dl(self.interp.ds_type).dataset[idx]
        res = ''
        for c, n in zip(da.cats, da.names[:len(da.cats)]):
          string = f'{da.classes[n][c]}'
          if string == 'True' or string == 'False':
            string += ';'
            res += string

          else:
            string = string[1:]
            res += string + ';'
        for c, n in zip(da.conts, da.names[len(da.cats):]):
          res += f'{c:.4f};'
        arr.append(res)
      f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
      for i, var in enumerate(self.interp.data.cont_names):
        f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
      f['Original'] = 'Original'
      dfarr.append(f)
      
      
      for j, x in enumerate(comb):
        arr = []
        for i, idx in enumerate(tl_idx):
          da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
          cl = int(cl)
          
          if classes[self.interp.pred_class[idx]] == comb[j][0] and classes[cl] == comb[j][1]:
            res = ''
            for c, n in zip(da.cats, da.names[:len(da.cats)]):
              string = f'{da.classes[n][c]}'
              if string == 'True' or string == 'False':
                string += ';'
                res += string
              else:
                string = string[1:]
                res += string + ';'
            for c, n in zip(da.conts, da.names[len(da.cats):]):
              res += f'{c:.4f};'
            arr.append(res)      
        f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
        for i, var in enumerate(self.interp.data.cont_names):
          f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
        f[str(x)] = str(x)
        dfarr.append(f)
      self.create_tabs(dfarr, cat_names)
