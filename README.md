# IR-211458294_206669707
Building a search engine for English Wikipedia
backend.py:
This module is responsible for all functionality that is needed in order to get relevant documents for a query.
It contains functions that are responsible for loading from bucket the pickle files that contain the title and body indexes, and several dictionaries.
It also contains the preprocessing functionality for given query text.
And for the must part, it contains functions that calculate cos similarity, and choose the top 100 ranked documents.

inverted_index_gcp.py:
This module contains the inverted index class, that has functions like reading a posting list of inverted index from bucket, writing a posting list of inverted index to bucket.
It also contains the MultiFileReader and MultiFileWriter classes.
Search_frontend.py:
This module contains the heart of our project- the search function, that gets a query, and using the functionality defined in the backend module, returns the top 100 ranked documents.
startup_script_gcp.sh:
A shell script that sets up the Compute Engine instance.
For running a query you should use:
http://34.30.169.5:8080/search?query=PUT_QUERY_HERE
