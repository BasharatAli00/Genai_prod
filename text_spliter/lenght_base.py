from langchain_text_splitters import CharacterTextSplitter

text="""
The Moon, Earth's only natural satellite, orbits our planet at about 384,400 kilometers away and is roughly one-quarter Earth's size. Its surface features ancient impact craters, dark volcanic plains called maria, and rugged highlands. Without an atmosphere, the Moon experiences extreme temperatures ranging from 127°C in sunlight to -173°C in shadow, and its landscape bears the scars of countless meteoroid impacts over billions of years.
The Moon's gravitational influence creates ocean tides and stabilizes Earth's axial tilt, helping moderate our climate. Its 29.5-day cycle has shaped human calendars and cultures throughout history. Since the Apollo 11 landing in 1969
"""

text_splitter=CharacterTextSplitter(chunk_size=100,chunk_overlap=0, separator="")
splited_text=text_splitter.split_text(text)
print(len(splited_text))
print(splited_text[0])
