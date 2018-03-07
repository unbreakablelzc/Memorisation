nb_samples_train = 15907
nb_samples_test = 5000

def balance():
    fo= open('data/memlog_200_2.txt', "r")
    f_train = open('data/memlog_2.txt', "w")
    f_test = open('data/memog_2_test.txt', 'w+')
    i = 1
    ling =0
    yi = 0

    test_ling = 0
    test_yi = 0

    for line in fo:
        list_line = line.split()
        list_line = list(map(int, list_line))
        # odd_numbered line
        if i % 2 == 1:
            flag_train = 0
            flag_test = 0
            if (list_line[0] == 0):
                if ling <= nb_samples_train:
                    f_train.write(line)
                    flag_train = 1

                else:
                    if test_ling <= nb_samples_test:
                        f_test.write(line)
                        flag_test = 1
                        test_ling += 1

                ling += 1
            else:
                if yi <= nb_samples_train:
                    f_train.write(line)
                    flag_train = 1
                else:
                    if test_yi <= nb_samples_test:
                        f_test.write(line)
                        flag_test = 1
                        test_yi += 1
                yi += 1

        # even_numbered line
        if i % 2 == 0:
            if(flag_train == 1):
                f_train.write(line)
            if(flag_test == 1):
                f_test.write(line)
            length = len(list_line)

        i = i + 1


if __name__ == "__main__":
    balance()

