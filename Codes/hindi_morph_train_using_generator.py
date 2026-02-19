""""Train Morphological Tagger using BiLSTM model and data generator using Keras framework."""
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser
# from gensim.models import KeyedVectors
from pickle import load
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
# from re import findall
# from re import S


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def createVectors(lines, word2Index, wordEmbeddings, char2Index, label2Index):
    """Create vectors from lines read from a file."""
    allSentenceVectors, sentenceVectors = list(), list()
    allLabelVectors = list()
    labelsForSent = list()
    charSequencesForWords, charSequencesForSentences = list(), list()
    maxSentenceLength, maxWordLength = 165, 18
    for line in lines:
        if line.strip():
            word, label = line.strip().split('\t')
            charSequenceForWord = [char2Index[char] if char in char2Index else len(
                char2Index) + 1 for char in word]
            charSequencesForWords.append(charSequenceForWord)
            charSequenceForWord = list()
            sentenceVectors.append(
                wordEmbeddings[word2Index[word]].tolist())
            labelsForSent.append(label2Index[label])
        else:
            if sentenceVectors:
                allSentenceVectors.append(sentenceVectors)
                sentenceVectors = list()
                allLabelVectors.append(labelsForSent)
                labelsForSent = list()
                charSequencesForSentences.append(charSequencesForWords)
                charSequencesForWords = list()
                charSequenceForWord = list()
    wholeCharSeqForData = list()
    finalSentenceVectors = pad_sequences(
        allSentenceVectors, padding='post', maxlen=maxSentenceLength)
    finalLabelSequences = pad_sequences(
        allLabelVectors, padding='post', maxlen=maxSentenceLength)
    allSentenceVectors, sentenceVectors = list(), list()
    charSequencesForWords = list()
    allLabelVectors = list()
    for seq in charSequencesForSentences:
        print(seq, 'Char SEQ')
        wholeCharSeqForData.append(pad_sequences(
            seq, maxlen=maxWordLength, padding='post'))
    charSequencesForSentences = list()
    finalCharSequences = pad_sequences(
        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
    print('Input Shape')
    print(finalCharSequences.shape, finalSentenceVectors.shape)
    print('SIZE OUTPUT')
    print(finalLabelSequences.shape)
    exit(1)
    return [finalCharSequences, finalSentenceVectors], finalLabelSequences


def createVectorsWithGenerator(lines, word2Index, wordEmbeddings, char2Index, label2Index):
    """Create vectors from lines read from a file."""
    allSentenceVectors, sentenceVectors = list(), list()
    allLabelVectors = list()
    labelsForSent = list()
    charSequencesForWords, charSequencesForSentences = list(), list()
    maxSentenceLength, maxWordLength = 165, 18
    while True:
        for line in lines:
            if line.strip():
                word, label = line.strip().split('\t')
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = list()
                sentenceVectors.append(
                    wordEmbeddings[word2Index[word]].tolist())
                labelsForSent.append(label2Index[label])
            else:
                if sentenceVectors:
                    allSentenceVectors.append(sentenceVectors)
                    sentenceVectors = list()
                    allLabelVectors.append(labelsForSent)
                    labelsForSent = list()
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = list()
                    charSequenceForWord = list()
                    wholeCharSeqForData = list()
                    finalSentenceVectors = pad_sequences(
                        allSentenceVectors, padding='post', maxlen=maxSentenceLength)
                    finalLabelSequences = pad_sequences(
                        allLabelVectors, padding='post', maxlen=maxSentenceLength)
                    allSentenceVectors, sentenceVectors = list(), list()
                    charSequencesForWords = list()
                    allLabelVectors = list()
                    for seq in charSequencesForSentences:
                        print(seq, 'Char SEQ')
                        wholeCharSeqForData.append(pad_sequences(
                            seq, maxlen=maxWordLength, padding='post'))
                    charSequencesForSentences = list()
                    finalCharSequences = pad_sequences(
                        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
                    print('Input Shape')
                    print(finalCharSequences.shape, finalSentenceVectors.shape)
                    print('SIZE OUTPUT')
                    print(finalLabelSequences.shape)
                    # exit(1)
                    finalLabelSequences = finalLabelSequences.reshape((-1, maxSentenceLength, 1))
                    yield [finalCharSequences, finalSentenceVectors], finalLabelSequences


def loadObjectToPickleFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, totalChars, trainData, trainLabels, totalLabels, weightFile, valData, valLabels, batchSize, epochs=1):
    """Train a model using BiLSTM."""
    embeddingLayer = Embedding(
        totalChars + 1, 50, mask_zero=True, input_length=maxWordLen, trainable=True, name='embedding-from-chars')
    charInput = Input(shape=(maxWordLen,), name='char-inputs')
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM
                               (50, return_sequences=False, dropout=0.3), merge_mode='sum', name='biLSTM_with_chars')(charEmbedding)
    charModel = Model(charInput, charOutput, name='char-model-for-repr')
    charSeq = Input(shape=(maxSentLen, maxWordLen), name='char-seq-for-sentence')
    charTD = TimeDistributed(charModel, name='td-for-chars')(charSeq)
    wordSeq = Input(shape=(maxSentLen, 200), name='word-input')
    merge = concatenate([wordSeq, charTD], axis=-1, name='concat-chars-word')
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 250), return_sequences=True, dropout=0.3), merge_mode='sum', name='biLSTM-for-words')(merge)
    wordSeqTDForLabel = TimeDistributed(Dense(totalLabels), name='td-for-outputs')(wordSeqLSTM)
    activationForLabel = Activation('softmax', name='activationForLcat')(wordSeqTDForLabel)
    finalModel = Model(
        inputs=[charSeq, wordSeq], outputs=activationForLabel, name='model-for-word-to-lcat')
    finalModel.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    print(finalModel.summary())
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    # batch_size = 16
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    finalModel.fit(trainData, trainLabels, epochs=epochs, validation_data=(valData, valLabels), callbacks=[checkpointCallback], batch_size=batchSize)
    return finalModel


def trainModelUsingBiLSTMAndGenerator(maxWordLen, maxSentLen, totalChars, trainGen, totalLabels, weightFile, valGen, batchSize, epochs=1):
    """Train a model using BiLSTM."""
    embeddingLayer = Embedding(
        totalChars + 1, 50, mask_zero=True, input_length=maxWordLen, trainable=True, name='embedding-from-chars')
    charInput = Input(shape=(maxWordLen,), name='char-inputs')
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM
                               (50, return_sequences=False, dropout=0.3), merge_mode='sum', name='biLSTM_with_chars')(charEmbedding)
    charModel = Model(charInput, charOutput, name='char-model-for-repr')
    charSeq = Input(shape=(maxSentLen, maxWordLen), name='char-seq-for-sentence')
    charTD = TimeDistributed(charModel, name='td-for-chars')(charSeq)
    wordSeq = Input(shape=(maxSentLen, 200), name='word-input')
    merge = concatenate([wordSeq, charTD], axis=-1, name='concat-chars-word')
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 250), return_sequences=True, dropout=0.3), merge_mode='sum', name='biLSTM-for-words')(merge)
    wordSeqTDForLabel = TimeDistributed(Dense(totalLabels), name='td-for-outputs')(wordSeqLSTM)
    activationForLabel = Activation('softmax', name='activationForLcat')(wordSeqTDForLabel)
    finalModel = Model(
        inputs=[charSeq, wordSeq], outputs=activationForLabel, name='model-for-word-to-lcat')
    finalModel.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    print(finalModel.summary())
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    # batch_size = 16
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    # finalModel.fit_generator(trainGen, steps_per_epoch=steps, epochs=epochs, validation_data=valGen, validation_steps=batchSize, callbacks=[checkpointCallback])
    finalModel.fit_generator(trainGen, epochs=epochs, validation_data=valGen, callbacks=[checkpointCallback], steps_per_epoch=2, validation_steps=2)
    return finalModel


def main():
    parser = ArgumentParser(
        description='Train the morph tagging model')
    parser.add_argument('--train', dest='tr',
                        help='Enter the input file containing morph labeled data')
    parser.add_argument('--val', dest='val', help='Enter the validation data')
    parser.add_argument('--mem', dest='mem',
                        help='enter memMapped file')
    parser.add_argument('--w2i', dest='w2i',
                        help='enter word2Index pickle file')
    parser.add_argument('--c2i', dest='c2i',
                        help='enter char2Index pickle file')
    parser.add_argument('--type', dest='type', help='enter the type of morph feature, use L for lcat, G for gender, N for number, P for person, C for case, V for vibhakti')
    parser.add_argument('--wt', dest='wt',
                        help='enter weight file')
    args = parser.parse_args()
    if not args.tr or not args.val or not args.mem or not args.w2i or not args.c2i or not args.wt:
        print(
            "Usage is python script_name --input input_file_name --w2v w2vecFile")
        exit(1)
    else:
        if args.type == 'L':
            index2Label = loadObjectToPickleFile('index-to-lcat-tb.pkl')
        elif args.type == 'G':
            index2Label = loadObjectToPickleFile('index-to-gender-tb.pkl')
        elif args.type == 'N':
            index2Label = loadObjectToPickleFile('index-to-number-tb.pkl')
        elif args.type == 'P':
            index2Label = loadObjectToPickleFile('index-to-person-tb.pkl')
        elif args.type == 'C':
            index2Label = loadObjectToPickleFile('index-to-case-tb.pkl')
        elif args.type == 'V':
            index2Label = loadObjectToPickleFile('index-to-vibh-tb.pkl')
        label2Index = createReverseIndex(index2Label)
        word2Index = loadObjectToPickleFile(args.w2i)
        char2Index = loadObjectToPickleFile(args.c2i)
        memoryMapped = np.memmap(args.mem, dtype='float32',
                                 mode='r', shape=(len(word2Index), 200))
        print('TOTAL CHARS', len(char2Index))
        print('WORD-VECTORS-SIZE', memoryMapped.shape)
        # epochs = 25
        epochs = 1
        batchSize = 2
        totalLabels = len(label2Index)
        maxWordLength, maxSentenceLength = 18, 165
        trainLines = read_lines_from_file(args.tr)
        # trainData, trainLabels = createVectors(
        #     trainLines, word2Index, memoryMapped, char2Index, label2Index)
        trainGen = createVectorsWithGenerator(
            trainLines, word2Index, memoryMapped, char2Index, label2Index)
        # for item in trainGen:
        #     print(item)
        # trainLabels = trainLabels.reshape((-1, maxSentenceLength, 1))
        valLines = read_lines_from_file(args.val)
        # valData, valLabels = createVectors(
        #     valLines, word2Index, memoryMapped, char2Index, label2Index)
        valGen = createVectorsWithGenerator(
            valLines, word2Index, memoryMapped, char2Index, label2Index)
        # valLabels = valLabels.reshape((-1, maxSentenceLength, 1))
        del memoryMapped
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        # biLSTM = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, len(char2Index), trainData, trainLabels, len(label2Index), args.wt, valData, valLabels, batchSize, epochs=epochs)
        biLSTM = trainModelUsingBiLSTMAndGenerator(maxWordLength, maxSentenceLength, len(char2Index), trainGen, len(label2Index), args.wt, valGen, batchSize, epochs=epochs)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    # tf.compat.v1.disable_eager_execution()
    config.gpu_options.allow_growth = True
#    sess = tf.compat.v1.Session(config=config)
#    sess.run(main)
    tf.compat.v1.experimental.output_all_intermediates(True)
    main()
