def slice_array(arr, box, s):
    '''
    returns a slice of the image according to the box coordinates
    '''
    i, j = box[0], box[1]
    sliced_array = arr[i:i + s, j:j + s, :]
    return sliced_array


def slices(fetch_img, s):
    '''
    Returns all of the slices for the given image
    '''
    slices = list()
    boolean = True
    j = 0
    while j + s + 1 in range(fetch_img.shape[1] - 1):
        i = 0
        while i + s + 1 in range(fetch_img.shape[0] - 1):
            box = [i, j]
            slices.append(slice_array(fetch_img, box, s))
            boolean = False
            i = i + s
        j = j + s
    i = fetch_img.shape[0] - (s + 1)
    j = 0
    if i > 0:
        while j + s + 1 in range(fetch_img.shape[1] - 1):
            box = [i, j]
            slices.append(slice_array(fetch_img, box, s))
            boolean = False
            j = j + s
    j = fetch_img.shape[1] - (s + 1)
    i = 0
    if j > 0:
        while i + s in range(fetch_img.shape[0] - 1):
            box = [i, j]
            slices.append(slice_array(fetch_img, box, s))
            boolean = False
            i = i + s
    if boolean:
        slices.append(fetch_img)
    return slices


def drive_slice(image, size):
    '''
    Drives the slicing function
    '''
    return slices(image, size)
