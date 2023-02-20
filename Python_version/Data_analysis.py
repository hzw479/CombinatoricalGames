datafile=open('Data','r')

data=datafile.readlines()[0].split(', ')

unique_data=set(data)
print(unique_data)
