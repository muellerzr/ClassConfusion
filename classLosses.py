import math
from fastai.vision import ClassificationInterpretation

def ClassLosses(interp:ClassificationInterpretation, k:float, class_1:str, class_2:str):
    c1, c2 = class_1, class_2
    val, idxs = interp.top_losses(len(interp.losses))
    tb1 = str(class_1 + ' x ' + class_2)
    tb2 = str(class_2 + ' x ' + class_1)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k/cols)
    
    classes = interp.data.classes
    
    vals = interp.most_confused()
    
    for x in range(len(vals)):
      if class_1 and class_2 in vals[x]:
        if vals[x][0] == class_1:
          rg1 = vals[x][2]
        if vals[x][0] == class_2:
          rg2 = vals[x][2]
  
  
    x1 = 0
    x2 = 0

    tb = widgets.TabBar([str(c1 + ' x ' + c2),
              str(c2 + ' x ' + c1)])
    with tb.output_to(tb1):
      fig1, axes1 = plt.subplots(rows, cols, figsize=(8,8))
      [axi.set_axis_off() for axi in axes1.ravel()]
      for i, idx in enumerate(idxs):
        if k < x1+1 or x1 > rg1:
          break
        da, cl = interp.data.dl(interp.ds_type).dataset[idx]
        row1 = (int)(x1 / cols)
        col1 = x1 % cols

        ix = int(cl)
        if str(cl) == c1 and str(classes[interp.pred_class[idx]]) == c2:
          da = image2np(da.data*255).astype(np.uint8)
          axes1[row1, col1].imshow(da)
          x1 += 1
      
      fig1.show()
    with tb.output_to(tb2):
        fig2, axes2 = plt.subplots(rows, cols, figsize=(8,8))
        [axi.set_axis_off() for axi in axes2.ravel()]
        for i, idx in enumerate(idxs):
          if k < x2+1 or x2 > rg2:
            break
          da, cl = interp.data.dl(interp.ds_type).dataset[idx]
          row2 = (int)(x2 / cols)
          col2 = x2 % cols

          ix = int(cl)
          if str(cl) == c2 and str(classes[interp.pred_class[idx]]) == c1:
            da = image2np(da.data*255).astype(np.uint8)
            axes2[row2, col2].imshow(da)
            x1 += 1
        
        fig2.show()
