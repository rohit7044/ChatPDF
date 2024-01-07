# ChatPDF
Chat with your PDF. Make summary. Ask relevant questions
This is a short project I did to explore the LLM and RAG domain. 
</br>Tested on Mistral 7B models.
### Please read the limitations at the end.

## Setup
1. Create environment</br>
``conda create -n chatPDF python=3.10 ``
2. Install poetry</br>
``pip install poetry``
3. Then run</br>
``poetry install``
4. Poetry will <b>not</b> install llama-cpp-python</br>Use ``pip install llama-cpp-python`` to install the CPU version. If you want to use CUDA, follow the steps in [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
5. Download the model and save it in the ``model`` directory 
6. You can use various models and embeddings. Just go to ``model.py`` and change the model and embeddings as you like.
7. Make sure to keep your PDF in ``Scanned_Documents``
8. Go to ``chat_PDF_Mistral7B.py`` or ``chat_PDF_OpenAI.py`` and update the prompt.



## Things to improve
1. Fix the vector database calling
2. Inefficient Architecture: Need to fix the database area. 
 </br>Loading the files in the database might be a better option. Might increase the speed. But if for only one time use, I think this architecture is fine.
If user wants to check on the data again in the future, the cold start approach will be an issue.
3. Metrics to check the text generation quality by using various RAG techniques. ``hit_rate`` and ``mrr``
4. All the code running on CPU. But still it's fast. Too lazy to reinstall ``llama_cpp_python``using CUBLAS.
5. Run the code using arguments
6. Run the code asynchronously
7. Sometimes the sentence is not complete proving that there's a limit on Mistral 7B.


## Weird things in the code
1. Used Mistral7B but llamaindex still needs OPENAI API keys to run some function like ``VectorStoreIndex``
2. So, reloading the database is a bit of challenge with this code because of the OPENAI requests even if I am not using OPENAI!!!