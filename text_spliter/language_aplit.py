from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
markuplanguage example
<p>hello</p>
<p>world</p>    
<p>hello</p>
<p>world</p>
    
"""

text_splitter=RecursiveCharacterTextSplitter.from_language(language="markdown",chunk_size=25,chunk_overlap=0)
splited_text=text_splitter.split_text(text)
print(len(splited_text))
print(f"The first chunk is: {splited_text[0]}")
print(f"The second chunk is: {splited_text[1]}")
# print(f"The third chunk is: {splited_text[2]}")
# print(f"The fourth chunk is: {splited_text[3]}")

print(splited_text)