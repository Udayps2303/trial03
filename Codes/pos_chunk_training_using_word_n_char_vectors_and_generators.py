from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser
from pickle import load
from keras.callbacks import ModelCheckpoint
# import numpy as np
import tensorflow as tf
import fasttext
from re import findall
from re import S


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return fileRead.readlines()


def createVectors(lines, wordEmbeddings, char2Index, pos2Index, chunk2Index):
    maxSentenceLength, maxWordLength = 150, 30
    allsentenceVectors = list()
    sentenceVectors = list()
    allPOSTagVectors, allChunkTagVectors, posTagsForSent, chunkTagsForSent = list(
    ), list(), list(), list()
    charSequencesForWords, charSequencesForSentences = list(), list()
    while True:
        for line in lines:
            if line.strip():
                word, posTag, chunkTag = line.strip().split('\t')
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = list()
                sentenceVectors.append(
                    wordEmbeddings.get_word_vector(word).tolist())
                posTagsForSent.append(pos2Index[posTag])
                chunkTagsForSent.append(chunk2Index[chunkTag])
            else:
                if len(sentenceVectors) > 0:
                    allsentenceVectors.append(sentenceVectors)
                    allPOSTagVectors.append(posTagsForSent)
                    allChunkTagVectors.append(chunkTagsForSent)
                    sentenceVectors, posTagsForSent, chunkTagsForSent = list(), list(), list()
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = list()
                    charSequenceForWord = list()
                    wholeCharSeqForData = list()
                    finalSentenceVectors = pad_sequences(
                        allsentenceVectors, padding='post', maxlen=maxSentenceLength)
                    finalPOSTagSequences = pad_sequences(
                        allPOSTagVectors, padding='post', maxlen=maxSentenceLength)
                    finalChunkTagSequences = pad_sequences(
                        allChunkTagVectors, padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = pad_sequences(charSequencesForSentences[0], maxlen=maxWordLength, padding='post')
                    # for charSeq in charSequencesForSentences:
                    #     wholeCharSeqForData.append(pad_sequences(
                    #         charSeq, maxlen=maxWordLength, padding='post'))
                    finalCharSequences = pad_sequences(
                        [wholeCharSeqForData], padding='post', maxlen=maxSentenceLength)
                    # print('FINAL CHAR SEQ', finalCharSequences.shape)
                    finalPOSTagSequences = finalPOSTagSequences.reshape((-1, maxSentenceLength, 1))
                    finalChunkTagSequences = finalChunkTagSequences.reshape((-1, maxSentenceLength, 1))
                    wholeCharSeqForData = list()
                    charSequencesForSentences = list()
                    allsentenceVectors = list()
                    allPOSTagVectors = list()
                    allChunkTagVectors = list()
                    yield [finalCharSequences, finalSentenceVectors], [finalPOSTagSequences, finalChunkTagSequences]


def loadObjectToPickleFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, trainGen, valGen, steps, valSteps, totalChars, totalPOS, totalChunks, weightFile, epochs=1):
    embeddingLayer = Embedding(
        totalChars + 1, 50, input_length=maxWordLen, trainable=True)
    charInput = Input(shape=(maxWordLen,))
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM
                               (50, return_sequences=False, dropout=0.3), merge_mode='sum')(charEmbedding)
    charModel = Model(charInput, charOutput)
    charSeq = Input(shape=(maxSentLen, maxWordLen))
#    charMask = Masking(mask_value=0)(charSeq)
    charTD = TimeDistributed(charModel)(charSeq)
    wordSeq = Input(shape=(maxSentLen, 300))
    merge = concatenate([wordSeq, charTD], axis=-1)
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 350), return_sequences=True, dropout=0.3), merge_mode='sum')(merge)
    wordSeqTDForPOS = TimeDistributed(Dense(totalPOS))(wordSeqLSTM)
    activationForPOS = Activation('softmax', name='activationForPOS')(wordSeqTDForPOS)
    wordSeqTDForChunk = TimeDistributed(Dense(totalChunks))(wordSeqLSTM)
    activationForChunk = Activation('softmax', name='activationForChunk')(wordSeqTDForChunk)
    finalModel = Model(
        inputs=[charSeq, wordSeq], outputs=[activationForPOS, activationForChunk])
    finalModel.compile(optimizer='adam',
                       loss={'activationForPOS': 'sparse_categorical_crossentropy',
                             'activationForChunk': 'sparse_categorical_crossentropy'},
                       metrics=['accuracy'])
    print(finalModel.summary())
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    finalModel.fit(trainGen,
                   steps_per_epoch=steps, epochs=epochs, callbacks=[checkpointCallback], validation_data=valGen, validation_steps=valSteps)
    return finalModel


def main():
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--train', dest='tr',
                        help='Enter the input file containing pos/chunk labeled data')
    parser.add_argument('--val', dest='val', help='Enter the validation data')
    parser.add_argument('--embed', dest='embed',
                        help='enter word embeddings file')
    parser.add_argument('--c2i', dest='c2i',
                        help='enter char2Index pickle file')
    parser.add_argument('--i2p', dest='i2p',
                        help='enter index2pos pickle file')
    parser.add_argument('--i2c', dest='i2c',
                        help='enter index2chunk pickle file')
    parser.add_argument('--wt', dest='wt',
                        help='enter weight file')
    parser.add_argument('--epoch', dest='epoch',
                        help='enter the no of epochs', type=int)
    args = parser.parse_args()
    if not args.tr or not args.val or not args.embed or not args.c2i or not args.i2p or not args.i2c or not args.wt or not args.epoch:
        print("Passed Arguments are not correct")
        exit(1)
    else:
        index2POS = loadObjectToPickleFile(args.i2p)
        index2Chunk = loadObjectToPickleFile(args.i2c)
        char2Index = loadObjectToPickleFile(args.c2i)
        print(char2Index)
        pos2Index = createReverseIndex(index2POS)
        chunk2Index = createReverseIndex(index2Chunk)
        wordEmbeddings = fasttext.load_model(args.embed)
        trainFileDesc = open(args.tr, 'r', encoding='utf-8')
        trainData = trainFileDesc.read() + '\n'
        trainFileDesc.close()
        totalSamples = len(findall('\n\n', trainData, S))
        print('Total Samples', totalSamples)
        batchSize = 16
        if totalSamples % batchSize == 0:
            steps = totalSamples // batchSize
        else:
            steps = totalSamples // batchSize + 1
        print('--TRAIN GEN--')
        trainLines = trainData.split('\n')
        trainGen = createVectors(
            trainLines, wordEmbeddings, char2Index, pos2Index, chunk2Index)
        valFileDesc = open(args.val, 'r', encoding='utf-8')
        valData = valFileDesc.read() + '\n'
        valFileDesc.close()
        totalValSamples = len(findall('\n\n', valData, S))
        print('Total Val Samples', totalValSamples)
        if totalValSamples % batchSize == 0:
            valSteps = totalValSamples // batchSize
        else:
            valSteps = totalValSamples // batchSize + 1
        print('--VAL GEN--')
        valLines = valData.split('\n')
        valGen = createVectors(
            valLines, wordEmbeddings, char2Index, pos2Index, chunk2Index)
        del wordEmbeddings
        maxWordLength, maxSentenceLength = 30, 150
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        biLSTM = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, trainGen, valGen, steps, valSteps,
                                       len(char2Index), len(pos2Index), len(chunk2Index), args.wt, args.epoch)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    # tf.compat.v1.disable_eager_execution()
    config.gpu_options.allow_growth = True
#    sess = tf.compat.v1.Session(config=config)
#    sess.run(main)
    main()
