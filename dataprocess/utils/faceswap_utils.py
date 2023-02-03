import numpy as np
import cv2


def AlphaBlend(foreground, background, alpha):
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    if len(alpha.shape) < 3:
        alpha = np.expand_dims(alpha, 2)

    # # Multiply the foreground with the alpha matte
    # foreground = cv2.multiply(alpha, foreground)

    # # Multiply the background with ( 1 - alpha )
    # background = cv2.multiply(1.0 - alpha, background)

    # # Add the masked foreground and background.
    # outImage = cv2.add(foreground, background)

    outImage = alpha * foreground + (1.-alpha) * background
    outImage = np.clip(outImage, 0, 255).astype(np.uint8)

    return outImage

# tutaj src to obraz, z ktorego piksele beda wklejane do obrazu dst
# feather amount to procent, sluzacy do sterowania wielkoscia obszaru, ktory bedzie poddany wagowaniu


def blendImages(src, dst, mask, featherAmount=0.1):
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
    maskPts = np.hstack(
        (maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    #hull = hull.astype(np.uint64)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        point = (maskPts[i, 0], maskPts[i, 1])
        dists[i] = cv2.pointPolygonTest(hull, point, True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0],
                                                                               maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    newMask = np.zeros_like(dst).astype(np.float32)
    newMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis]

    return composedImg, newMask

# uwaga, tutaj src to obraz, z ktorego brany bedzie kolor


def colorTransfer(src_, dst_, mask):
    src = dst_
    dst = src_
    transferredDst = np.copy(dst)
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst
    return transferredDst
