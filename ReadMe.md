# ClassConfusion

Class Confusion was was designed to help extrapolate your models decisions through visuals such as graphs or confusion matrices that go more in-depth than the standard plot_confusion_matrix. Class Confusion can be used with both Tabular and Image classification models

To utilize this function, input in the ClassificationInterpretation object as well as a list of classes you want to examine:
```python3
from ClassLosses import *
ClassLosses(interp, classList)
```

You can also pass in direct class combinations you want to see as well, just make them a list of tuples as such:

```python3
comboList = [('<50k', '>=50k')]
ClassLosses(interp, comboList, is_ordered=True)
```

Please read the Documentation for a guide to how to utilize this function.

Some example outputs:
![](https://camo.githubusercontent.com/dc2f4b6e86db5e41274b60e605de25dd3a29ee27/68747470733a2f2f692e696d6775722e636f6d2f6a41453642566d2e706e67)

![](https://camo.githubusercontent.com/cefb9ee9dd7ed469afff8b899040a8330ca043df/68747470733a2f2f692e696d6775722e636f6d2f695555537032412e706e67)
