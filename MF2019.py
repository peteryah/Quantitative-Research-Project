
def data_to_local(X, Y, dist):
    temp = list(X.keys())
    with open('symbols.txt', 'w') as symbols:
        for i in temp:
            symbols.write('%s\n' % i)
    for i in temp:
        filenameX = 'dataset/X/' + i + " X.json"
        filenameY = 'dataset/Y/' + i + ' Y.txt'
        filenameD = 'dataset/D/' + i + ' D.json'
        with open(filenameX, 'w') as f:

            out = pd.DataFrame(X[i]).to_json()
            f.write(out)
        with open(filenameY, 'w') as f:
            for k in Y[i]:
                f.write('%s\n' % str(k))
        with open(filenameD, 'w') as f:
            out = pd.DataFrame(dist[i]).to_json()
            f.write(out)

    return
