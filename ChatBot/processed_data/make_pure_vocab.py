








if __name__ == '__main__':
    with open('vocab.txt','r',encoding='utf-8') as fr:
        fw = open('vocab_pure.txt','w',encoding='utf-8')
        for item in fr:
            if not (item.strip().startswith("##") or item.strip().startswith("[unused")):
                fw.write(item)
        fw.close()
