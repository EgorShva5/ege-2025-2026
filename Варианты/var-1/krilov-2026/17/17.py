text = open('text.txt', mode='r', encoding='UTF-8')

all_nums = [int(i) for i in text]
dvuh_nums = [i for i in all_nums if abs(i/100) < 1 and abs(i) >= 10]
minimum = min(dvuh_nums)
maximum = max(dvuh_nums)

uk = 0

cnt = 0
m = 0
for b in range(0, len(all_nums)-2):
    ch = (all_nums[b], all_nums[b+1], all_nums[b+2])
    dvuh_nums = [i for i in ch if abs(i/100) < 1 and abs(i) >= 10]
    
    if len(dvuh_nums) >= 2 and sum(ch) > minimum+maximum:
        m = max(m, sum(ch))
        cnt += 1

print(cnt, m)