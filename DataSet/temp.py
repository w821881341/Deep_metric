import os
# import os.path as osp
root = '/opt/intern/users/xunwang/DataSet/Products/train'

for i, paths in enumerate(os.walk(root)):
    if i%1024 == 1:
        print(paths)
        # print(i)
    if i == 0:
        continue
    # print(paths)
    with open('/opt/intern/users/xunwang/DataSet/Products/train.txt', 'a') as f:
        lines_ = ['{} {}\n'.format(os.path.join('train', paths[0], img), i) for img in paths[2]]
        f.writelines(lines_)
    # break