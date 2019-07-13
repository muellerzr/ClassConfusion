# ClassLosses

Class losses is a widget (currently only for Google Colab) that helps visualize what our model's weaknesses by providing visualizations and examining the losses between two or more class combinations. For image models, this is done by providing the overall image. For tabular models, the tool will provide variable-by-variable analysis of the trends found within the data.

To utilize this function, input in the ClassificationInterpretation object as well as a list of classes you want to examine:
```
from ClassLosses import *
ClassLosses(interp, classList)
```

An example of an image problem:

![](https://i.imgur.com/UzxFkzc.png)

An example of a tabular problem:

![](https://i.imgur.com/7S9vjsQ.png)
![](https://i.imgur.com/oN1mXR7.png)
