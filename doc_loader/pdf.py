from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('grok_report.pdf')
doc=loader.load()
print(type(doc))
print(doc)
print(doc[0].page_content)  
print(doc[1].page_content)
