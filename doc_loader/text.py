from langchain_community.document_loaders import TextLoader

loader=TextLoader('docker.txt')
doc=loader.load()
print("\n") 
print(type(doc))
print("\n") 
print(len(doc))
print("\n") 
print(doc[0].page_content)


