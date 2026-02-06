from typing import TypedDict


class person(TypedDict):
    name : str
    age : int

new_person: person={
    'name' : 'yousuf',
    'age'  : '25'
}

print(new_person)