import argparse
import os
from string import punctuation


symbols = set(punctuation) - set(';,"।?!')
print(symbols)


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return fileRead.readlines()


def updateIncorrectChunkTags(lines, level=1):
    updatedLines = list()
    prevLabel, prevType = '', ''
    for line in lines:
        if line.strip():
            if level:
                token, posTag, chunkTag = line.strip().split('\t')
            else:
                chunkTag = line.strip()
            if level:
                if token in symbols:
                    posTag = 'RD_SYM'
                    line = '\t'.join([token, posTag, chunkTag]) + '\n'
                if token in ';,"।?!':
                    posTag = 'RD_PUNC'
                    line = '\t'.join([token, posTag, chunkTag]) + '\n'
            chunkLabel, chunkType = chunkTag.split('-')
            if not prevLabel and not prevType:
                if chunkLabel == 'I':
                    if level:
                        updatedLines.append(token + '\t' + posTag + '\t' + 'B-' + chunkType + '\n')
                    else:
                        updatedLines.append('B-' + chunkType + '\n')
                    prevLabel = 'B'
                else:
                    updatedLines.append(line)
                    prevLabel = chunkLabel
                prevType = chunkType
            else:
                if chunkLabel != prevLabel and chunkType == prevType:
                    updatedLines.append(line)
                    prevLabel = chunkLabel
                    prevType = chunkType
                elif chunkType != prevType and chunkLabel == 'I':
                    if level:
                        updatedLines.append(token + '\t' + posTag + '\t' + 'B-' + chunkType + '\n')
                    else:
                        updatedLines.append('B-' + chunkType + '\n')
                    prevLabel = 'B'
                    prevType = chunkType
                else:
                    updatedLines.append(line)
                    prevLabel = chunkLabel
                    prevType = chunkType
        else:
            updatedLines.append(line)
            prevLabel, prevType = '', ''
    return updatedLines


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write(''.join(dataList))


def readFilesFromFolderUpdateIncorrectChunksAndWrite(inputFolderPath, outputFolderPath, level=1):
    if os.path.isdir(inputFolderPath):
        for root, dirs, files in os.walk(inputFolderPath):
            inputFilePaths = [os.path.join(root, fl) for fl in files]
            outputFilePaths = [os.path.join(outputFolderPath, fl) for fl in files]
        for indexFile, inputPath in enumerate(inputFilePaths):
            updatedLines = updateIncorrectChunkTags(readLinesFromFile(inputPath), level)
            writeListToFile(outputFilePaths[indexFile], updatedLines)
    else:
        updatedLines = updateIncorrectChunkTags(readLinesFromFile(inputFolderPath), level)
        writeListToFile(outputFolderPath, updatedLines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input folder path')
    parser.add_argument('--output', dest='out', help='Enter the output folder path')
    args = parser.parse_args()
    # directoryPath = args.inp[os.path.abspath(args.inp).rfind('/') + 1:]
    if os.path.isdir(args.inp):
        if not os.path.isdir(args.out):
            os.mkdir(args.out)
    readFilesFromFolderUpdateIncorrectChunksAndWrite(args.inp, args.out, level=0)
