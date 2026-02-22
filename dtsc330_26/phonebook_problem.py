#two phone books
#firstname lastname 
#how do we connect data when there are differences in the data?
#create features then classifers
#HW create training data for this model
#phone books there 
#can you find anyway where 
#you dont have to connect everyone in phone book 1 to phonebook 2 explicitly
# Authors DF - first name, last name, initials, affiliations
# Grantees - same columns
import random
import pandas as pd
import numpy as np

#Make a phonebook with just first and last names for now.
#Make a second phonebook based off the first one with some errors.

#code: 
#normal phonebook
first_names = [
    "Michael", "Sarah", "John", "Emily", "David",
    "Ashley", "Daniel", "Jessica", "Matthew", "Amanda"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Miller", "Davis", "Garcia", "Wilson", "Taylor"
]

N = 20  # number of rows

phonebook_1 = pd.DataFrame({
    "first_name": [random.choice(first_names) for _ in range(N)],
    "last_name": [random.choice(last_names) for _ in range(N)]
})

#corrupted phonebook
#randomly pick an entry in the messed up phone book dataset.
#randomly pick either first name, last name, or both.
#to pick a random spot in the name to randomly remove, add, or replace a letter.
phonebook_2 = phonebook_1.copy()

alphabet_list = ['A','B','C','D','E','F','G','H','I','J','K','L',
    'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def error(name):
    if len(name) == 0:
        return name
    
    action = random.choice(["remove", "add", "replace"])
    pos = random.randint(0, len(name)-1)
    
    if action == "remove" and len(name) > 1:
        name = name[:pos] + name[pos+1:]
        
    elif action == "add":
        letter = random.choice(alphabet_list)
        name = name[:pos] + letter + name[pos:]
        
    elif action == "replace":
        letter = random.choice(alphabet_list)
        name = name[:pos] + letter + name[pos+1:]
    
    return name

num_errors = int(N * 0.6)  # corrupt 60% of rows

rows_to_modify = random.sample(range(N), num_errors)

for row in rows_to_modify:
    
    modify_choice = random.choice(["first", "last", "both"])
    
    if modify_choice == "first":
        phonebook_2.loc[row, "first_name"] = error(
            phonebook_2.loc[row, "first_name"]
        )
        
    elif modify_choice == "last":
        phonebook_2.loc[row, "last_name"] = error(
            phonebook_2.loc[row, "last_name"]
        )
        
    else:  # both
        phonebook_2.loc[row, "first_name"] = error(
            phonebook_2.loc[row, "first_name"]
        )
        phonebook_2.loc[row, "last_name"] = error(
            phonebook_2.loc[row, "last_name"]
        )

print("Clean Phonebook (A)")
print(phonebook_1)

print("\nMessy Phonebook (B)")
print(phonebook_2)