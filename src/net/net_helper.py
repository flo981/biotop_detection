def pyramid(image, scale=1.5, minSize=(20, 20)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        if w==0:
            w=1
            print("Excepten w:0->1")
            break
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def write_csv(output_dict):
    print("Write output .csv file...")
    out = open('out.csv', 'w')
    out.write("Biotop;")
    out.write("percentage;")
    out.write("nPatches;")
    out.write('\n')
    for row in output_dict:
        out.write('%s;' % row)
        out.write('%d;' % output_dict[row]['percentage'])
        out.write('%d;' % output_dict[row]['nPatches'])
        out.write('\n')
    out.close()
