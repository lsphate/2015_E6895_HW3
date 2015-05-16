with open("data.txt") as f:
    arr = []
    for line in f:
        arr.append(float(line))
    print arr
