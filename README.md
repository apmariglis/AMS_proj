##### Install dependencies

```
$ poetry install
```

##### Start a shell within the virtual environment

```
$poetry shell
```

##### Transform training data from xlsx to csv

```
$ python src/classifier/train_xlsx_2_csv.py --input resources/library.xlsx --output resources/library.csv --sheet Sheet1
```

##### Train the model

Ex. (~6 mins to execute) using only m/z peaks:[29 30 31 43 44 55 57 60 73 82 91]

```
$ python src/classifier/train_sparse_multinomial_w_elasticnet.py --csv resources/library.csv --model resources/model_11_feat.joblib --features 29 30 31 43 44 55 57 60 73 82 91 --quiet
```

Ex. (~7 mins to execute) using only m/z peaks:[29 30 31 43 44 55 57 60 73 82 84 91 115]

```
$ time python src/classifier/train_sparse_multinomial_w_elasticnet.py --csv resources/library.csv --model resources/model_13_feat.joblib --features 29 30 31 43 44 55 57 60 73 82 84 91 115 --quiet
```

##### Transform prediction data from xlsx to csv 

```
$ python src/classifier/predict_xlsx_2_csv.py --input resources/RusanenEtAl_synthetic.xlsx --output resources/RusanenEtAl_synthetic.csv --sheet X
```

##### Predict with the generated model

```
$ python src/classifier/predict.py --csv resources/RusanenEtAl_synthetic.csv --model resources/model_11_feat.joblib --outdir resources/ 
```

```
$ python src/classifier/predict.py --csv resources/RusanenEtAl_synthetic.csv --model resources/model_13_feat.joblib --outdir resources/ 
```

##### Exit the poetry shell

```
$ exit
```

