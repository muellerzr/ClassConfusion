import math
import pandas as pd
import matplotlib.pyplot as plt

from itertools import permutations
from google.colab import widgets
from fastai.vision import ClassificationInterpretation

class ClassLosses():
  "Plot the most confused datapoints and statistics for your misses. \nPass in a `interp` object and a list of classes to look at."
  def __init__(self, interp:ClassificationInterpretation, classlist:list, is_ordered:bool=False):
    self.interp = interp
    if str(type(interp.learn.data)) == "<class 'fastai.tabular.data.TabularDataBunch'>":
      if interp.learn.data.train_ds.x.cont_names != []: 
        self.means = interp.learn.data.train_ds.x.processor[0].procs[2].means
        self.stds = interp.learn.data.train_ds.x.processor[0].procs[2].stds
    self.is_ordered = is_ordered
    self.show_losses(classlist)
    
    
  def create_graphs(self, df_list:list, cat_names:list):
    print('Variable Distrobution:')
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
            ttl = str.join('', df_list[j].columns[-1])
            if j == 0:
              title = ttl + ' ' + tbnames[i]+' distrobution'
            else:
              title = 'Misclassified ' + ttl + ' ' + tbnames[i]+' distrobution'
            fig = vals.plot(kind='bar', title= title, rot=30, ax=ax[j])
          else:
            vals = df_list[j][tab].astype(float)
            vals = vals * self.stds[tab] + self.means[tab]
            ttl = str.join('', df_list[j].columns[-1])
            if j == 0:
              title = ttl + ' ' + tbnames[i] + ' distrobution'
            else:
              title = 'Misclassified ' + ttl + ' ' + tbnames[i]+' distrobution'
            
            axs = vals.plot(kind='hist', ax=ax[j])
            vals.plot(kind='kde', ax=axs, title = title, secondary_y=True)
            
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
        f[var] = f[var].apply(lambda x: float(x) * norm.stds[var] + norm.means[var])
      f['Original'] = 'Original'
      dfarr.append(f)
      
      
      for j, x in enumerate(comb):
        arr = []
        for i, idx in enumerate(tl_idx):
          da, cl = interp.data.dl(interp.ds_type).dataset[idx]
          cl = int(cl)
          
          if classes[interp.pred_class[idx]] == comb[j][0] and classes[cl] == comb[j][1]:
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
          f[var] = f[var].apply(lambda x: float(x) * norm.stds[var] + norm.means[var])
        f[str(x)] = str(x)
        dfarr.append(f)
      
      self.create_graphs(dfarr, cat_names)
