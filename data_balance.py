def balance():
    fo= open('data/memlog_200_1.txt', "r")
    f = open('data/memlog.txt', "w")
    i = 1
    ling =0
    yi = 0
    for line in fo:
        list_line = line.split()
        list_line = list(map(int, list_line))
        # odd_numbered line
        if i % 2 == 1:
            if (list_line[0] == 0):
                ling += 1
                flag = 0
            else:
                yi += 1
                flag = 1
           # even_numbered line
        if i % 2 == 0:
            length = len(list_line)
        i = i + 1
        if yi <= 8504 and flag == 1:
            f.write(line)
        if ling <=8504 and flag == 0:
            f.write(line)
    print(ling)
    print(yi)
if __name__ == "__main__":
    balance()

