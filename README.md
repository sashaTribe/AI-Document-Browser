# AI-Document-Browser
In this project I attempted to create an AI Chat where you can upload your own PDF and query that information

Although functional, there are a few bugs that still need resolving:
- Although not a bug, there is code and imports not being used, this is still to be fixed but I need to check exacly what I can and cannot delete.
- There is a weird loop happening where it will attempt to print the first query more than once, still investigating.
- When you first ask a question it gives you an answer alongside an error. Once you upload a document and then ask a question this does not happen, still investigating.

What are the next steps to improve this app?
- FAISS Stores the informartion localy, perhaps a way to know what PDFs have been ingested already as to not get repeats.
- Add a way to catch errors when uploading the same file twice.
- Webscraping, a way to automatically ingest PDFs from the web.
- Wikipedia Ingestor, as well as PDFs I believe it could be of use to be able to add information from Wikipedia directly for an added layer of context. (Use case for this particular project would be to maintain an updated list of UK officials)
