text = open('17.txt', mode='r', encoding='UTF-8')

all_nums = [int(i) for i in text]
okon_seven = [i for i in all_nums if str(i)[-2:] == '70']
maximum = max(okon_seven)

cnt =0
max_summa = 0
for i in range(len(all_nums)-2):
    a_s = (all_nums[i], all_nums[i+1], all_nums[i+2])

    if a_s[0] >= 0 and a_s[1] >= 0 and a_s[2] >= 0 and sum(a_s) <= maximum:
        max_summa = max(max_summa, sum(a_s))
        cnt += 1

print(cnt, max_summa)