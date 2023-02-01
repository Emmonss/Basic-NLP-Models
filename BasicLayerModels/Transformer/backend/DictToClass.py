import json
class DictToClass(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':
    txt = {"Success": True, "ErrorCode": None, "Message": None,
           "Data": None, "Data555": [1, 2, 3, 4, 5]}
    TXT = DictToClass(**txt)
    print(TXT.Success)
    print(TXT.Data555)
