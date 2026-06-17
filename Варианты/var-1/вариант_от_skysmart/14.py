for x in '0123456789abcd':
    if (int(f'{x}38D6', 14) + int(f'{x}C624',14))%5498==0:
        print(x, (int(f'{x}38D6', 14) + int(f'{x}C624',14))/5498)
        break