from math import sqrt

xVals = [1, 2, 10, 5, 6, 4, 8, 0]
yVals = [3, 5, 4, 1, 2, 7, 3, 6]

hierarchy = []

for i in range(len(xVals)):
    steps = []
    for j in range(len(yVals)):
        value = sqrt((xVals[i]-xVals[j])**2 + (yVals[i]-yVals[j])**2)
        steps.append(value)
    hierarchy.append(steps)

print(hierarchy)