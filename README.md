# Code of IAAS Framework
## Run This
```shell
cd src
python run.py
```
## Output File Explanation
- mdoel.db: detail record of all searched model
- each searched model contains:
  - prediction result in test data
  - transformation table
  - model parameters of type ```.pth```
  - loss curve while training