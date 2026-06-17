text = open('text.txt', encoding='UTF-8').readlines()

cnt = 0
for i in text:
    i.strip('\n')
    nums = i.split()
    int_nums = [int(b) for b in nums]
    sorted_nums = sorted(int_nums)
    
    if len(nums) == len(set(nums)):
        if ((sorted_nums[0] + sorted_nums[-1])/2) > ((sum(int_nums)-(sorted_nums[0] + sorted_nums[-1]))/3):
            cnt += 1 

print(cnt)