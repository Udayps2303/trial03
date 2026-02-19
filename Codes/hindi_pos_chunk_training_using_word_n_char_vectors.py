from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser
from pickle import load
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf


def createVectorsForDataAndLabels(inputFilePath, word2Index, wordEmbeddings, char2Index, pos2Index, chunk2Index):
    maxSentenceLength, maxWordLength = 0, 0
    allsentenceVectors = list()
    sentenceVectors = list()
    allPOSTagVectors, allChunkTagVectors, posTagsForSent, chunkTagsForSent = list(
    ), list(), list(), list()
    charSequencesForWords, charSequencesForSentences = list(), list()
    with open(inputFilePath) as test:
        for line in test:
            if line.strip():
                word, posTag, chunkTag = line.strip().split('\t')
                if maxWordLength < len(word):
                    maxWordLength = len(word)
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                if charSequenceForWord:
                    charSequencesForWords.append(charSequenceForWord)
                sentenceVectors.append(
                    wordEmbeddings[word2Index[word]].tolist())
                posTagsForSent.append(pos2Index[posTag])
                chunkTagsForSent.append(chunk2Index[chunkTag])
            else:
                if maxSentenceLength < len(sentenceVectors):
                    maxSentenceLength = len(sentenceVectors)
                allsentenceVectors.append(sentenceVectors)
                allPOSTagVectors.append(posTagsForSent)
                allChunkTagVectors.append(chunkTagsForSent)
                sentenceVectors, posTagsForSent, chunkTagsForSent = list(), list(), list()
                charSequencesForSentences.append(charSequencesForWords)
                charSequencesForWords = list()
    wholeCharSeqForData = list()
    maxSentenceLength, maxWordLength = 165, 18
    finalSentenceVectors = pad_sequences(
        allsentenceVectors, padding='post', maxlen=maxSentenceLength)
    finalPOSTagSequences = pad_sequences(
        allPOSTagVectors, padding='post', maxlen=maxSentenceLength)
    finalChunkTagSequences = pad_sequences(
        allChunkTagVectors, padding='post', maxlen=maxSentenceLength)
    for charSeq in charSequencesForSentences:
        wholeCharSeqForData.append(pad_sequences(
            charSeq, maxlen=maxWordLength, padding='post'))
    finalCharSequences = pad_sequences(
        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
    return finalCharSequences, finalSentenceVectors, finalPOSTagSequences, finalChunkTagSequences


def loadObjectToPickleFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, sentenceVectors, charSequences, posTags, chunkTags, totalChars, totalPOS, totalChunks, weightFile, valData):
    print(charSequences.shape, sentenceVectors.shape)
    embeddingLayer = Embedding(
        totalChars + 1, 50, mask_zero=True, input_length=maxWordLen, trainable=True)
    charInput = Input(shape=(maxWordLen,))
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM
                               (50, return_sequences=False, dropout=0.3), merge_mode='sum')(charEmbedding)
    charModel = Model(charInput, charOutput)
    charSeq = Input(shape=(maxSentLen, maxWordLen))
#    charMask = Masking(mask_value=0)(charSeq)
    charTD = TimeDistributed(charModel)(charSeq)
    wordSeq = Input(shape=(maxSentLen, 200))
    merge = concatenate([wordSeq, charTD], axis=-1)
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 250), return_sequences=True, dropout=0.3), merge_mode='sum')(merge)
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
    posTags = posTags.reshape((-1, maxSentLen, 1))
    chunkTags = chunkTags.reshape((-1, maxSentLen, 1))
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    batch_size = 16
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    finalModel.fit([charSequences, sentenceVectors], [posTags, chunkTags],
                   batch_size=batch_size, epochs=1, callbacks=[checkpointCallback], validation_data=([valData[0], valData[1]], [valData[2], valData[3]]))
    return finalModel


def main():
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--train', dest='inp',
                        help='Enter the input file containing pos/chunk labeled data')
    parser.add_argument('--val', dest='val', help='Enter the validation data')
    parser.add_argument('--mem', dest='mem',
                        help='enter memMapped file')
    parser.add_argument('--w2i', dest='w2i',
                        help='enter word2Index pickle file')
    parser.add_argument('--c2i', dest='c2i',
                        help='enter char2Index pickle file')
    parser.add_argument('--wt', dest='wt',
                        help='enter weight file')
    args = parser.parse_args()
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config)
    #K.set_session(sess)
    if not args.inp or not args.val or not args.mem or not args.w2i or not args.c2i or not args.wt:
        print(
            "Usage is python script_name --input input_file_name --w2v w2vecFile")
        exit(1)
    else:
        #index2POS = loadObjectToPickleFile('pos-tags-to-index-tb-ilmt.pkl')
        index2POS = loadObjectToPickleFile('pos-tags-to-index-tb-jud.pkl')
        #index2Chunk = loadObjectToPickleFile('chunk-tags-to-index-tb-ilmt.pkl')
        index2Chunk = loadObjectToPickleFile('chunk-tags-to-index-tb-jud.pkl')
        word2Index = loadObjectToPickleFile(args.w2i)
        char2Index = loadObjectToPickleFile(args.c2i)
        print(char2Index)
        pos2Index = createReverseIndex(index2POS)
        chunk2Index = createReverseIndex(index2Chunk)
        memoryMapped = np.memmap(args.mem, dtype='float32',
                                 mode='r', shape=(len(word2Index), 200))
        print(memoryMapped.shape)
        finalCharSequences, finalSentenceVectors, finalPOSTagSequences, finalChunkTagSequences = createVectorsForDataAndLabels(
            args.inp, word2Index, memoryMapped, char2Index, pos2Index, chunk2Index)
        # del word2vecModel
        validationData = createVectorsForDataAndLabels(
                            args.val, word2Index, memoryMapped, char2Index, pos2Index, chunk2Index)
        del memoryMapped
        maxWordLength, maxSentenceLength = 18, 165
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        biLSTM = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, finalSentenceVectors, finalCharSequences,
                                       finalPOSTagSequences, finalChunkTagSequences, len(char2Index), len(pos2Index), len(chunk2Index), args.wt, validationData)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    tf.compat.v1.disable_eager_execution()
    config.gpu_options.allow_growth = True
#    sess = tf.compat.v1.Session(config=config)
#    sess.run(main)
    main()
