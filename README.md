Alpha I Model Store
===================

This is the repository of alpha-i models

Install
-------
```bash
$ conda create -n aimodels python=3.6
$ source activate aimodels
```

On GPU machines:
```bash
$ pip install -r gpu-requirements-gpu.txt

```

On CPU machines:
```bash
$ pip install -r cpu-requirements.txt

```

Running tests
-------------

```bash
$ pip install -r dev-requirements.txt
$ pytest tests/ -s --disable-warnings #to make the output cleaner

```
