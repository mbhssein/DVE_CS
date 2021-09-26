

def getBlocks(img, k):
    """ Break the image up into kxk blocks. Crop if necessary."""

    # Throw an error if not grayscale.
    if len(img.shape) != 2:
        print ("Image is not grayscale. Returning empty block list.")
        return []
    
    blocks = []
    n_vert = img.shape[0] / k
    n_horiz = img.shape[1] / k

    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            blocks.append(img[i*k:(i+1)*k, j*k:(j+1)*k])

    return blocks
