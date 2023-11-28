# DomainDefectDataset
A dataset of hand-annotated emergent domain defects in 1D elementary cellular automata to benchmark test (or train) automated detection algorithms


## example

A random sample
```python
python synthetic_data_generator.py 
```

More control?
```python
python synthetic_data_generator.py --width 100 --time 200 --max_phase 3 --n_domains 3 
```
![](example.png)

Specify the exact patterns and locations of each domain
```python
python synthetic_data_generator.py --domain_pattern 1 0 --domain_pattern 0 1 --domain_centre 10 10 --domain_centre 50 50  
```

## Todo:
- store data sample in json (include base64 encoding of image and labelled image)
- allow number of samples to be generated from CLI