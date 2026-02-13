from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path='monthly_load.csv')
doc=loader.load()
print(len(doc))
print(doc)
print(doc[1].metadata)
