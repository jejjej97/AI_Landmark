# 라벨 읽기

def label_read():
    f = open('landmark_label.csv', 'r', encoding='utf-8')
    lst = f.readlines()
    print(lst)
    class_list = lst[0]
    f.close()
    print(class_list)
    print(type(class_list))

    result = class_list.split(',')
    print(result)

    result.pop()
    print(result)
    print(type(result))

    return result