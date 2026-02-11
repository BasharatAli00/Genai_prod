from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
The Moon, Earth's only natural satellite, orbits our planet at about 384,400 kilometers away and is roughly one-quarter Earth's size. Its surface features ancient impact craters, dark volcanic plains called maria, and rugged highlands. Without an atmosphere, the Moon experiences extreme temperatures ranging from 127°C in sunlight to -173°C in shadow, and its landscape bears the scars of countless meteoroid impacts over billions of years.
The Moon's gravitational influence creates ocean tides and stabilizes Earth's axial tilt, helping moderate our climate. Its 29.5-day cycle has shaped human calendars and cultures throughout history. Since the Apollo 11 landing in 1969
"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0, separators=["\n\n", "\n", ".", " "])
splited_text=text_splitter.split_text(text)
print(len(splited_text))
print(f"The first chunk is: {splited_text[0]}")
print(f"The second chunk is: {splited_text[1]}")
# print(f"The third chunk is: {splited_text[2]}")
# print(f"The fourth chunk is: {splited_text[3]}")

print(splited_text)