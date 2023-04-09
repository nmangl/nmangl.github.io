#캐글 타이타닉 

라이브러리를 불러오고
'train.csv','test.csv' 파일도 불러온다

```python
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_28276\3017818251.py in <module>
    ----> 1 import pandas as pd
          2 import numpy as np
          3 
          4 train = pd.read_csv('train.csv')
          5 test = pd.read_csv('test.csv')
    

    ModuleNotFoundError: No module named 'pandas'



```python
train.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_28276\642956413.py in <module>
    ----> 1 train.head()
    

    NameError: name 'train' is not defined



```python

```
