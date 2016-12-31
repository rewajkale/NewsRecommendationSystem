
Data from: BBC News

Make sure you have **gensim** & **nltk** installed in your python path, this repo depeneds on these libraries.

* Build model
```
python /path-to-repo/LDAModel_English.py --dir /path-to-repo/BBCNews
```
Model building should take minutes to run, depends on how large your corpus is.
* After the model is generated and saved, now try to get some recommendations: 

```
python /path-to-repo/application.py
```
And the flask application will run

Note: The entire BBC data is not uploaded but you can scrape it and get it
