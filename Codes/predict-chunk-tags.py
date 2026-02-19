from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser
import numpy as np
from keras.models import load_model
from pickle import load
import tensorflow as tf


def loadObjectToPickleFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def loadWordVectorsFromFile(filePath):
    return KeyedVectors.load_word2vec_format(
        filePath, binary=True, unicode_errors='ignore')


def vectorizeData(lines, wordEmbeddings, char2Index, vectorLength, maxWordLength, maxSentenceLength):
    sentenceVectors, allSentenceVectors, charSequencesForWords, charSequencesForSentences = list(
    ), list(), list(), list()
    for line in lines:
        line = line.strip()
        if line:
            if line in wordEmbeddings:
                sentenceVectors.append(wordEmbeddings[line].tolist())
            else:
                sentenceVectors.append(np.zeros(vectorLength).tolist())
            charSequenceForWord = [char2Index[char] if char in char2Index else len(
                char2Index)for char in line]
            if charSequenceForWord:
                charSequencesForWords.append(charSequenceForWord)
        else:
            if len(sentenceVectors) > maxSentenceLength:
                multiples = len(sentenceVectors) // maxSentenceLength
                for i in range(multiples):
                    allSentenceVectors.append(
                        sentenceVectors[i * maxSentenceLength: (i + 1) * maxSentenceLength])
                    charSequencesForSentences.append(
                        charSequencesForWords[i * maxSentenceLength: (i + 1) * maxSentenceLength])
                if len(sentenceVectors) % maxSentenceLength:
                    allSentenceVectors.append(
                        sentenceVectors[(i + 1) * maxSentenceLength:])
                    charSequencesForSentences.append(
                        charSequencesForWords[i * maxSentenceLength: (i + 1) * maxSentenceLength])
            else:
                allSentenceVectors.append(sentenceVectors)
                charSequencesForSentences.append(charSequencesForWords)
            sentenceVectors, charSequencesForWords = list(), list()
    if sentenceVectors and charSequencesForWords:
        allSentenceVectors.append(sentenceVectors)
        charSequencesForSentences.append(charSequencesForWords)
        sentenceVectors, charSequencesForWords = list(), list()
    wholeCharSeqForData = list()
    finalSentenceVectors = pad_sequences(
        allSentenceVectors, padding='post', maxlen=maxSentenceLength)
    for charSeq in charSequencesForSentences:
        wholeCharSeqForData.append(pad_sequences(
            charSeq, maxlen=maxWordLength, padding='post'))
    finalCharSequences = pad_sequences(
        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
    return finalSentenceVectors, finalCharSequences


def findPredictedTagsForSample(chunkTags, sampleLen, indexToChunk):
    predictedTags = ''
    for index in range(sampleLen):
        predictedTags += indexToChunk[np.argmax(chunkTags[index])] + '\n'
    return predictedTags


def predictChunkTags(trainedModel, sentenceVectors, charSeqVectors, indexToChunk, lines, maxSentenceLength):
    chunkTags = trainedModel.predict(
        [charSeqVectors, sentenceVectors])
    sentenceLengths, greaterDict = sentenceLengthsInLines(
        lines, maxSentenceLength)
    predictedTags = ''
    for indexSample in range(chunkTags.shape[0]):
        predictedTags += findPredictedTagsForSample(
            chunkTags[indexSample], sentenceLengths[indexSample], indexToChunk)
        predictedTags += '\n'
    return predictedTags, greaterDict


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return fileRead.readlines()


def sentenceLengthsInLines(lines, maxSentenceLength):
    lengths = list()
    sentLen = 0
    greaterDict = dict()
    index = 0
    for line in lines:
        if line.strip():
            sentLen += 1
        else:
            if sentLen > maxSentenceLength:
                greaterDict[index] = sentLen
                for i in range(sentLen // maxSentenceLength):
                    lengths.append(maxSentenceLength)
                lengths = lengths if not (
                    sentLen % maxSentenceLength) else lengths + [sentLen % maxSentenceLength]
            else:
                lengths.append(sentLen)
            sentLen = 0
            index += 1
    if sentLen > 0:
        if sentLen > maxSentenceLength:
            greaterDict[index] = sentLen
            for i in range(sentLen // maxSentenceLength):
                lengths.append(maxSentenceLength)
            lengths = lengths if not (
                sentLen % maxSentenceLength) else lengths + [sentLen % maxSentenceLength]
        else:
            lengths.append(sentLen)
    sentLen = 0
    return lengths, greaterDict


def mergeSeparatedData(predictedTags, maxSentenceLength, greaterDict):
    updatedTags, tempTags = list(), list()
    nextLen = 0
    index, flag = 0, 0
    for tag in predictedTags.split('\n'):
        if tag.strip():
            tempTags.append(tag)
        else:
            if index in greaterDict and not flag:
                # print('TRUE')
                nextLen = greaterDict[index]
                # print('NEXT', nextLen)
                flag = 1
            elif tempTags and nextLen == len(tempTags) and flag:
                # print('IND', index)
                updatedTags.append('\n'.join(tempTags) + '\n\n')
                nextLen = 0
                index += 1
                tempTags = list()
                flag = 0
            else:
                # print('IN ELSE')
                if tempTags:
                    updatedTags.append('\n'.join(tempTags) + '\n\n')
                    tempTags = list()
                    index += 1
    return ''.join(updatedTags)


def writeDataToFile(data, filePath):
    with open(filePath, 'w') as fileWrite:
        fileWrite.write(data + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input file')
    parser.add_argument('--w2v', dest='w2v', help='Enter the word2vec file')
    parser.add_argument('--c2i', dest='c2i',
                        help='Enter the char to index pickle file')
    parser.add_argument('--model', dest='model',
                        help='Enter the trained model')
    parser.add_argument('--output', dest='out',
                        help='Enter the output path where the predictions will be written to')
    args = parser.parse_args()
    lines = readLinesFromFile(args.inp)
    word2vecModel = loadWordVectorsFromFile(args.w2v)
    char2Index = loadObjectToPickleFile(args.c2i)
    indexToChunk = loadObjectToPickleFile('chunk-tags-to-index-tb-ilmt.pkl')
    trainedModel = load_model(args.model)
    maxWordLength = 18
    # 18, 162 for General / treebank and Judiciary Mixed data
    # 16, 118 for General data or treebank
    # 18, 116 for new_articles from treebank
    maxSentenceLength = 165
    finalSentenceVectors, finalCharSequences = vectorizeData(
        lines, word2vecModel, char2Index, 200, maxWordLength, maxSentenceLength)
    del word2vecModel
    pred_tags = trainedModel.predict([finalCharSequences, finalSentenceVectors])
    #exit(1)
    print(finalSentenceVectors.shape, 'SV', 'CV', finalCharSequences.shape)
    predictedTags, greaterDict = predictChunkTags(trainedModel, finalSentenceVectors,
                                    finalCharSequences, indexToChunk, lines, maxSentenceLength)
    updatedTags = mergeSeparatedData(predictedTags, maxSentenceLength, greaterDict)
    writeDataToFile(updatedTags, args.out)
    # writeDataToFile(predictedTags, args.out)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    tf.compat.v1.disable_eager_execution()
    config.gpu_options.allow_growth = True
    main()
