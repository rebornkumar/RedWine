
# redWine


```python
import numpy as np
import pandas as pd
```


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn import preprocessing
```


```python
from sklearn.ensemble import RandomForestRegressor
```


```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
```


```python
from sklearn.metrics import mean_squared_error, r2_score
```


```python
from sklearn.externals import joblib
```


```python
data = pd.read_csv("winequality-red.csv")
```


```python
print data.head()
```

      fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
    0   7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5                                                                                                                     
    1   7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5                                                                                                                     
    2  7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;...                                                                                                                     
    3  11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58...                                                                                                                     
    4   7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5                                                                                                                     
    


```python
data = pd.read_csv("winequality-red.csv",sep=';')
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.880</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.760</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.280</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.4</td>
      <td>0.660</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.075</td>
      <td>13.0</td>
      <td>40.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.9</td>
      <td>0.600</td>
      <td>0.06</td>
      <td>1.6</td>
      <td>0.069</td>
      <td>15.0</td>
      <td>59.0</td>
      <td>0.99640</td>
      <td>3.30</td>
      <td>0.46</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.3</td>
      <td>0.650</td>
      <td>0.00</td>
      <td>1.2</td>
      <td>0.065</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>0.99460</td>
      <td>3.39</td>
      <td>0.47</td>
      <td>10.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.8</td>
      <td>0.580</td>
      <td>0.02</td>
      <td>2.0</td>
      <td>0.073</td>
      <td>9.0</td>
      <td>18.0</td>
      <td>0.99680</td>
      <td>3.36</td>
      <td>0.57</td>
      <td>9.5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.5</td>
      <td>0.500</td>
      <td>0.36</td>
      <td>6.1</td>
      <td>0.071</td>
      <td>17.0</td>
      <td>102.0</td>
      <td>0.99780</td>
      <td>3.35</td>
      <td>0.80</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.7</td>
      <td>0.580</td>
      <td>0.08</td>
      <td>1.8</td>
      <td>0.097</td>
      <td>15.0</td>
      <td>65.0</td>
      <td>0.99590</td>
      <td>3.28</td>
      <td>0.54</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7.5</td>
      <td>0.500</td>
      <td>0.36</td>
      <td>6.1</td>
      <td>0.071</td>
      <td>17.0</td>
      <td>102.0</td>
      <td>0.99780</td>
      <td>3.35</td>
      <td>0.80</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.6</td>
      <td>0.615</td>
      <td>0.00</td>
      <td>1.6</td>
      <td>0.089</td>
      <td>16.0</td>
      <td>59.0</td>
      <td>0.99430</td>
      <td>3.58</td>
      <td>0.52</td>
      <td>9.9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7.8</td>
      <td>0.610</td>
      <td>0.29</td>
      <td>1.6</td>
      <td>0.114</td>
      <td>9.0</td>
      <td>29.0</td>
      <td>0.99740</td>
      <td>3.26</td>
      <td>1.56</td>
      <td>9.1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8.9</td>
      <td>0.620</td>
      <td>0.18</td>
      <td>3.8</td>
      <td>0.176</td>
      <td>52.0</td>
      <td>145.0</td>
      <td>0.99860</td>
      <td>3.16</td>
      <td>0.88</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.9</td>
      <td>0.620</td>
      <td>0.19</td>
      <td>3.9</td>
      <td>0.170</td>
      <td>51.0</td>
      <td>148.0</td>
      <td>0.99860</td>
      <td>3.17</td>
      <td>0.93</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.5</td>
      <td>0.280</td>
      <td>0.56</td>
      <td>1.8</td>
      <td>0.092</td>
      <td>35.0</td>
      <td>103.0</td>
      <td>0.99690</td>
      <td>3.30</td>
      <td>0.75</td>
      <td>10.5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.1</td>
      <td>0.560</td>
      <td>0.28</td>
      <td>1.7</td>
      <td>0.368</td>
      <td>16.0</td>
      <td>56.0</td>
      <td>0.99680</td>
      <td>3.11</td>
      <td>1.28</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7.4</td>
      <td>0.590</td>
      <td>0.08</td>
      <td>4.4</td>
      <td>0.086</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>0.99740</td>
      <td>3.38</td>
      <td>0.50</td>
      <td>9.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.9</td>
      <td>0.320</td>
      <td>0.51</td>
      <td>1.8</td>
      <td>0.341</td>
      <td>17.0</td>
      <td>56.0</td>
      <td>0.99690</td>
      <td>3.04</td>
      <td>1.08</td>
      <td>9.2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8.9</td>
      <td>0.220</td>
      <td>0.48</td>
      <td>1.8</td>
      <td>0.077</td>
      <td>29.0</td>
      <td>60.0</td>
      <td>0.99680</td>
      <td>3.39</td>
      <td>0.53</td>
      <td>9.4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7.6</td>
      <td>0.390</td>
      <td>0.31</td>
      <td>2.3</td>
      <td>0.082</td>
      <td>23.0</td>
      <td>71.0</td>
      <td>0.99820</td>
      <td>3.52</td>
      <td>0.65</td>
      <td>9.7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7.9</td>
      <td>0.430</td>
      <td>0.21</td>
      <td>1.6</td>
      <td>0.106</td>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.99660</td>
      <td>3.17</td>
      <td>0.91</td>
      <td>9.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8.5</td>
      <td>0.490</td>
      <td>0.11</td>
      <td>2.3</td>
      <td>0.084</td>
      <td>9.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.17</td>
      <td>0.53</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6.9</td>
      <td>0.400</td>
      <td>0.14</td>
      <td>2.4</td>
      <td>0.085</td>
      <td>21.0</td>
      <td>40.0</td>
      <td>0.99680</td>
      <td>3.43</td>
      <td>0.63</td>
      <td>9.7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6.3</td>
      <td>0.390</td>
      <td>0.16</td>
      <td>1.4</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>0.99550</td>
      <td>3.34</td>
      <td>0.56</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7.6</td>
      <td>0.410</td>
      <td>0.24</td>
      <td>1.8</td>
      <td>0.080</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>0.99620</td>
      <td>3.28</td>
      <td>0.59</td>
      <td>9.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7.9</td>
      <td>0.430</td>
      <td>0.21</td>
      <td>1.6</td>
      <td>0.106</td>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.99660</td>
      <td>3.17</td>
      <td>0.91</td>
      <td>9.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7.1</td>
      <td>0.710</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.080</td>
      <td>14.0</td>
      <td>35.0</td>
      <td>0.99720</td>
      <td>3.47</td>
      <td>0.55</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7.8</td>
      <td>0.645</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>0.082</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>0.99640</td>
      <td>3.38</td>
      <td>0.59</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>6.2</td>
      <td>0.510</td>
      <td>0.14</td>
      <td>1.9</td>
      <td>0.056</td>
      <td>15.0</td>
      <td>34.0</td>
      <td>0.99396</td>
      <td>3.48</td>
      <td>0.57</td>
      <td>11.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>6.4</td>
      <td>0.360</td>
      <td>0.53</td>
      <td>2.2</td>
      <td>0.230</td>
      <td>19.0</td>
      <td>35.0</td>
      <td>0.99340</td>
      <td>3.37</td>
      <td>0.93</td>
      <td>12.4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>6.4</td>
      <td>0.380</td>
      <td>0.14</td>
      <td>2.2</td>
      <td>0.038</td>
      <td>15.0</td>
      <td>25.0</td>
      <td>0.99514</td>
      <td>3.44</td>
      <td>0.65</td>
      <td>11.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>7.3</td>
      <td>0.690</td>
      <td>0.32</td>
      <td>2.2</td>
      <td>0.069</td>
      <td>35.0</td>
      <td>104.0</td>
      <td>0.99632</td>
      <td>3.33</td>
      <td>0.51</td>
      <td>9.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1573</th>
      <td>6.0</td>
      <td>0.580</td>
      <td>0.20</td>
      <td>2.4</td>
      <td>0.075</td>
      <td>15.0</td>
      <td>50.0</td>
      <td>0.99467</td>
      <td>3.58</td>
      <td>0.67</td>
      <td>12.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1574</th>
      <td>5.6</td>
      <td>0.310</td>
      <td>0.78</td>
      <td>13.9</td>
      <td>0.074</td>
      <td>23.0</td>
      <td>92.0</td>
      <td>0.99677</td>
      <td>3.39</td>
      <td>0.48</td>
      <td>10.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1575</th>
      <td>7.5</td>
      <td>0.520</td>
      <td>0.40</td>
      <td>2.2</td>
      <td>0.060</td>
      <td>12.0</td>
      <td>20.0</td>
      <td>0.99474</td>
      <td>3.26</td>
      <td>0.64</td>
      <td>11.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>8.0</td>
      <td>0.300</td>
      <td>0.63</td>
      <td>1.6</td>
      <td>0.081</td>
      <td>16.0</td>
      <td>29.0</td>
      <td>0.99588</td>
      <td>3.30</td>
      <td>0.78</td>
      <td>10.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>6.2</td>
      <td>0.700</td>
      <td>0.15</td>
      <td>5.1</td>
      <td>0.076</td>
      <td>13.0</td>
      <td>27.0</td>
      <td>0.99622</td>
      <td>3.54</td>
      <td>0.60</td>
      <td>11.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1578</th>
      <td>6.8</td>
      <td>0.670</td>
      <td>0.15</td>
      <td>1.8</td>
      <td>0.118</td>
      <td>13.0</td>
      <td>20.0</td>
      <td>0.99540</td>
      <td>3.42</td>
      <td>0.67</td>
      <td>11.3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1579</th>
      <td>6.2</td>
      <td>0.560</td>
      <td>0.09</td>
      <td>1.7</td>
      <td>0.053</td>
      <td>24.0</td>
      <td>32.0</td>
      <td>0.99402</td>
      <td>3.54</td>
      <td>0.60</td>
      <td>11.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>7.4</td>
      <td>0.350</td>
      <td>0.33</td>
      <td>2.4</td>
      <td>0.068</td>
      <td>9.0</td>
      <td>26.0</td>
      <td>0.99470</td>
      <td>3.36</td>
      <td>0.60</td>
      <td>11.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>6.2</td>
      <td>0.560</td>
      <td>0.09</td>
      <td>1.7</td>
      <td>0.053</td>
      <td>24.0</td>
      <td>32.0</td>
      <td>0.99402</td>
      <td>3.54</td>
      <td>0.60</td>
      <td>11.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1582</th>
      <td>6.1</td>
      <td>0.715</td>
      <td>0.10</td>
      <td>2.6</td>
      <td>0.053</td>
      <td>13.0</td>
      <td>27.0</td>
      <td>0.99362</td>
      <td>3.57</td>
      <td>0.50</td>
      <td>11.9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>6.2</td>
      <td>0.460</td>
      <td>0.29</td>
      <td>2.1</td>
      <td>0.074</td>
      <td>32.0</td>
      <td>98.0</td>
      <td>0.99578</td>
      <td>3.33</td>
      <td>0.62</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1584</th>
      <td>6.7</td>
      <td>0.320</td>
      <td>0.44</td>
      <td>2.4</td>
      <td>0.061</td>
      <td>24.0</td>
      <td>34.0</td>
      <td>0.99484</td>
      <td>3.29</td>
      <td>0.80</td>
      <td>11.6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1585</th>
      <td>7.2</td>
      <td>0.390</td>
      <td>0.44</td>
      <td>2.6</td>
      <td>0.066</td>
      <td>22.0</td>
      <td>48.0</td>
      <td>0.99494</td>
      <td>3.30</td>
      <td>0.84</td>
      <td>11.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1586</th>
      <td>7.5</td>
      <td>0.310</td>
      <td>0.41</td>
      <td>2.4</td>
      <td>0.065</td>
      <td>34.0</td>
      <td>60.0</td>
      <td>0.99492</td>
      <td>3.34</td>
      <td>0.85</td>
      <td>11.4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1587</th>
      <td>5.8</td>
      <td>0.610</td>
      <td>0.11</td>
      <td>1.8</td>
      <td>0.066</td>
      <td>18.0</td>
      <td>28.0</td>
      <td>0.99483</td>
      <td>3.55</td>
      <td>0.66</td>
      <td>10.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1588</th>
      <td>7.2</td>
      <td>0.660</td>
      <td>0.33</td>
      <td>2.5</td>
      <td>0.068</td>
      <td>34.0</td>
      <td>102.0</td>
      <td>0.99414</td>
      <td>3.27</td>
      <td>0.78</td>
      <td>12.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>6.6</td>
      <td>0.725</td>
      <td>0.20</td>
      <td>7.8</td>
      <td>0.073</td>
      <td>29.0</td>
      <td>79.0</td>
      <td>0.99770</td>
      <td>3.29</td>
      <td>0.54</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>6.3</td>
      <td>0.550</td>
      <td>0.15</td>
      <td>1.8</td>
      <td>0.077</td>
      <td>26.0</td>
      <td>35.0</td>
      <td>0.99314</td>
      <td>3.32</td>
      <td>0.82</td>
      <td>11.6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1591</th>
      <td>5.4</td>
      <td>0.740</td>
      <td>0.09</td>
      <td>1.7</td>
      <td>0.089</td>
      <td>16.0</td>
      <td>26.0</td>
      <td>0.99402</td>
      <td>3.67</td>
      <td>0.56</td>
      <td>11.6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1592</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>6.8</td>
      <td>0.620</td>
      <td>0.08</td>
      <td>1.9</td>
      <td>0.068</td>
      <td>28.0</td>
      <td>38.0</td>
      <td>0.99651</td>
      <td>3.42</td>
      <td>0.82</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>6.2</td>
      <td>0.600</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.090</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99490</td>
      <td>3.45</td>
      <td>0.58</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>5.9</td>
      <td>0.550</td>
      <td>0.10</td>
      <td>2.2</td>
      <td>0.062</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>0.99512</td>
      <td>3.52</td>
      <td>0.76</td>
      <td>11.2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1597</th>
      <td>5.9</td>
      <td>0.645</td>
      <td>0.12</td>
      <td>2.0</td>
      <td>0.075</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99547</td>
      <td>3.57</td>
      <td>0.71</td>
      <td>10.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1598</th>
      <td>6.0</td>
      <td>0.310</td>
      <td>0.47</td>
      <td>3.6</td>
      <td>0.067</td>
      <td>18.0</td>
      <td>42.0</td>
      <td>0.99549</td>
      <td>3.39</td>
      <td>0.66</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>1599 rows × 12 columns</p>
</div>




```python
print data.head()
```

       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0            7.4              0.70         0.00             1.9      0.076   
    1            7.8              0.88         0.00             2.6      0.098   
    2            7.8              0.76         0.04             2.3      0.092   
    3           11.2              0.28         0.56             1.9      0.075   
    4            7.4              0.70         0.00             1.9      0.076   
    
       free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                 11.0                  34.0   0.9978  3.51       0.56   
    1                 25.0                  67.0   0.9968  3.20       0.68   
    2                 15.0                  54.0   0.9970  3.26       0.65   
    3                 17.0                  60.0   0.9980  3.16       0.58   
    4                 11.0                  34.0   0.9978  3.51       0.56   
    
       alcohol  quality  
    0      9.4        5  
    1      9.8        5  
    2      9.8        5  
    3      9.8        6  
    4      9.4        5  
    


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print data.shape
```

    (1599, 12)
    


```python
Y=data.quality
X=data.drop('quality',axis=1)
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123,stratify=Y)

```


```python
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
```


```python
print X_train_scaled.mean(axis=0)
print "\n"
print X_train_scaled.std(axis=0)

```

    [ 1.16664562e-16 -3.05550043e-17 -8.47206937e-17 -2.22218213e-17
      2.22218213e-17 -6.38877362e-17 -4.16659149e-18 -2.54439854e-15
     -8.70817622e-16 -4.08325966e-16 -1.17220107e-15]
    
    
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    


```python
X_test_scaled = scaler.transform(X_test)
```


```python
print X_test_scaled.mean(axis=0)
print "\n"
print X_test_scaled.std(axis=0)
```

    [ 0.02776704  0.02592492 -0.03078587 -0.03137977 -0.00471876 -0.04413827
     -0.02414174 -0.00293273 -0.00467444 -0.10894663  0.01043391]
    
    
    [1.02160495 1.00135689 0.97456598 0.91099054 0.86716698 0.94193125
     1.03673213 1.03145119 0.95734849 0.83829505 1.0286218 ]
    


```python
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
```


```python
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
```


```python
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, Y_train)
```




    GridSearchCV(cv=10, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decr...mators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'randomforestregressor__max_depth': [None, 5, 3, 1], 'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
print clf.best_params_
```

    {'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'log2'}
    


```python
print clf.refit
```

    True
    


```python
Y_pred = clf.predict(X_test)
```


```python
print r2_score(Y_test, Y_pred)
```

    0.4739352261032751
    


```python
print mean_squared_error(Y_test, Y_pred)
```

    0.339455625
    


```python
joblib.dump(clf, 'rf_regressor.pkl')
```




    ['rf_regressor.pkl']




```python
clf2 = joblib.load('rf_regressor.pkl')
```


```python
clf2.predict(X_test)
```




    array([6.57, 5.76, 5.04, 5.6 , 6.12, 5.5 , 4.93, 4.86, 5.01, 6.07, 5.24,
           5.81, 5.81, 5.08, 5.83, 5.59, 6.6 , 5.78, 5.76, 6.98, 5.5 , 5.64,
           5.08, 6.03, 5.93, 5.04, 5.43, 5.18, 5.79, 6.01, 5.91, 6.4 , 6.  ,
           5.07, 4.9 , 5.93, 5.04, 6.1 , 5.03, 6.09, 4.91, 5.9 , 6.81, 5.1 ,
           6.25, 5.52, 5.47, 5.47, 5.06, 6.45, 6.07, 5.42, 5.72, 5.12, 5.62,
           5.72, 5.28, 5.27, 5.03, 5.3 , 5.35, 5.24, 5.01, 5.89, 5.77, 5.21,
           6.41, 5.04, 5.19, 6.64, 5.74, 5.59, 5.06, 5.01, 5.48, 5.98, 5.3 ,
           5.11, 5.23, 5.29, 6.29, 5.53, 6.25, 6.35, 5.06, 6.06, 6.46, 6.31,
           5.51, 5.83, 5.87, 5.34, 6.35, 5.62, 5.66, 5.74, 6.72, 6.73, 5.47,
           6.66, 5.13, 5.53, 5.11, 6.53, 5.08, 4.76, 5.65, 5.03, 5.73, 5.95,
           5.91, 5.51, 6.02, 5.34, 5.24, 5.25, 5.92, 5.09, 4.96, 5.94, 5.82,
           5.13, 5.81, 6.12, 5.29, 5.45, 5.23, 5.9 , 5.42, 5.34, 5.83, 6.29,
           5.15, 5.28, 5.08, 6.51, 5.05, 5.16, 6.63, 5.45, 5.19, 5.09, 5.62,
           6.01, 5.4 , 5.36, 5.08, 6.4 , 5.79, 5.15, 5.62, 5.16, 4.79, 5.02,
           5.25, 5.95, 5.33, 5.73, 5.77, 5.25, 5.56, 5.13, 5.31, 5.91, 5.07,
           6.01, 5.17, 5.31, 5.64, 5.17, 6.17, 5.09, 5.65, 5.02, 5.55, 5.42,
           5.  , 5.46, 5.49, 5.01, 6.03, 5.6 , 5.02, 4.99, 5.17, 6.21, 5.25,
           5.67, 5.35, 4.85, 5.48, 6.66, 5.81, 5.87, 5.58, 5.18, 5.45, 5.07,
           6.16, 4.86, 6.27, 5.07, 5.17, 5.26, 6.74, 6.08, 5.33, 5.22, 5.42,
           5.88, 5.84, 5.77, 5.99, 6.28, 5.81, 5.91, 5.25, 5.27, 5.71, 5.26,
           5.21, 6.16, 6.12, 5.56, 5.79, 5.92, 5.53, 6.13, 5.46, 5.74, 5.33,
           5.52, 6.22, 5.71, 4.97, 4.45, 6.68, 6.47, 6.31, 5.17, 5.36, 5.56,
           5.46, 6.3 , 6.08, 5.16, 5.18, 5.2 , 5.22, 6.26, 5.17, 5.04, 5.29,
           5.12, 5.87, 6.42, 5.67, 5.4 , 5.4 , 6.46, 5.55, 6.13, 5.26, 5.24,
           5.72, 5.83, 5.83, 5.56, 5.42, 5.08, 5.74, 5.6 , 6.41, 6.04, 5.65,
           4.77, 5.93, 6.51, 6.1 , 5.44, 5.57, 5.28, 5.3 , 6.07, 6.89, 5.2 ,
           6.48, 5.84, 5.22, 5.4 , 5.64, 5.1 , 5.29, 6.28, 5.82, 5.93, 6.07,
           5.93, 5.36, 5.77, 5.54, 6.08, 5.62, 6.94, 6.94, 5.88, 6.32, 5.07,
           5.37, 5.93, 5.23, 5.32, 5.93, 6.7 , 6.33, 5.35, 5.56, 5.7 , 6.08,
           5.46])


