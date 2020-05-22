import pandas as pd

# creditCardData = pd.read_csv('creditcard.csv')

# lst = list(creditCardData.columns.values)
# print(lst)


# Function to create formula to for regression model
def createFormula(dataColumnsLst):
    result = dataColumnsLst[-1] + " ~ "
    count = 0
    for label in dataColumnsLst[:-1]:
        if count == 0:
            result = result + label
        else:
            result = result + "+" + label
        count += 1

    return str(result)

# print(createFormula(lst))
