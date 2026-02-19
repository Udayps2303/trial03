"""Train Hindi Morphological Tagger using BiLSTM model and data generator using Keras framework."""
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
from re import findall
from re import S


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def createVectors(lines, word2Index, wordEmbeddings, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index):
    allSentenceVectors, sentenceVectors = list(), list()
    allLcatVectors, allGenderVectors, allNumberVectors, allPersonVectors, allCaseVectors, allVibhVectors = list(), list(), list(), list(), list(), list()
    lcatsForSent, gendersForSent, numbersForSent, personsForSent, casesForSent, vibhsForSent = list(), list(), list(), list(), list(), list()
    charSequencesForWords, charSequencesForSentences = list(), list()
    maxSentenceLength, maxWordLength = 165, 18
    while True:
        for line in lines:
            if line.strip():
                word, lcat, gender, number, person, case, vibh = line.strip().split('\t')
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = list()
                # print('CWs', charSequencesForWords)
                sentenceVectors.append(
                    wordEmbeddings[word2Index[word]].tolist())
                lcatsForSent.append(lcat2Index[lcat])
                gendersForSent.append(gender2Index[gender])
                numbersForSent.append(number2Index[number])
                personsForSent.append(person2Index[person])
                casesForSent.append(case2Index[case])
                vibhsForSent.append(vibh2Index[vibh])
            else:
                if sentenceVectors:
                    allSentenceVectors.append(sentenceVectors)
                    # print(len(sentenceVectors), 'SENT')
                    # print(len(allSentenceVectors), 'A')
                    sentenceVectors = list()
                    # print(lcatsForSent, 'LCS')
                    allLcatVectors.append(lcatsForSent)
                    allGenderVectors.append(gendersForSent)
                    allNumberVectors.append(numbersForSent)
                    allPersonVectors.append(personsForSent)
                    allCaseVectors.append(casesForSent)
                    allVibhVectors.append(vibhsForSent)
                    lcatsForSent, gendersForSent, numbersForSent, personsForSent, casesForSent, vibhsForSent = list(), list(), list(), list(), list(), list()
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = list()
                    # print(charSequencesForSentences, 'CSS', id(charSequencesForSentences), id(charSequencesForWords))
                    charSequenceForWord = list()
                    # print('1st IF')
                    wholeCharSeqForData = list()
                    finalSentenceVectors = pad_sequences(
                        allSentenceVectors, padding='post', maxlen=maxSentenceLength)
                    finalLcatSequences = pad_sequences(
                        allLcatVectors, padding='post', maxlen=maxSentenceLength)
                    finalLcatSequences = finalLcatSequences.reshape((-1, maxSentenceLength, 1))
                    finalGenderSequences = pad_sequences(
                        allGenderVectors, padding='post', maxlen=maxSentenceLength)
                    finalGenderSequences = finalGenderSequences.reshape((-1, maxSentenceLength, 1))
                    finalNumberSequences = pad_sequences(
                        allNumberVectors, padding='post', maxlen=maxSentenceLength)
                    finalNumberSequences = finalNumberSequences.reshape((-1, maxSentenceLength, 1))
                    finalPersonSequences = pad_sequences(
                        allPersonVectors, padding='post', maxlen=maxSentenceLength)
                    finalPersonSequences = finalPersonSequences.reshape((-1, maxSentenceLength, 1))
                    finalCaseSequences = pad_sequences(
                        allCaseVectors, padding='post', maxlen=maxSentenceLength)
                    finalCaseSequences = finalCaseSequences.reshape((-1, maxSentenceLength, 1))
                    finalVibhSequences = pad_sequences(
                        allVibhVectors, padding='post', maxlen=maxSentenceLength)
                    finalVibhSequences = finalVibhSequences.reshape((-1, maxSentenceLength, 1))
                    allSentenceVectors, sentenceVectors = list(), list()
                    charSequencesForWords = list()
                    allLcatVectors, allGenderVectors, allNumberVectors, allPersonVectors, allCaseVectors, allVibhVectors = list(), list(), list(), list(), list(), list()
                    for seq in charSequencesForSentences:
                        # print(seq, 'Char SEQ')
                        wholeCharSeqForData.append(pad_sequences(
                            seq, maxlen=maxWordLength, padding='post'))
                    charSequencesForSentences = list()
                    finalCharSequences = pad_sequences(
                        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = list()
                    allSentenceVectors = list()
                    yield [finalCharSequences, finalSentenceVectors], [finalLcatSequences, finalGenderSequences, finalNumberSequences, finalPersonSequences, finalCaseSequences, finalVibhSequences]


def loadObjectToPickleFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, totalChars, trainGen, totalLcats, totalGenders, totalNumbers, totalPersons, totalCases, totalVibhs, weightFile, valGen, steps, valSteps, epochs=1):
    # print(totalLcats, totalGenders, totalNumbers, totalPersons, totalCases, totalVibhs, 'MORPH')
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
    wordSeq = Input(shape=(maxSentLen, 200))
    merge = concatenate([wordSeq, charTD], axis=-1)
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 250), return_sequences=True, dropout=0.3), merge_mode='sum')(merge)
    wordSeqTDForLcat = TimeDistributed(Dense(totalLcats))(wordSeqLSTM)
    activationForLcat = Activation('softmax', name='activationForLcat')(wordSeqTDForLcat)
    wordSeqTDForGender = TimeDistributed(Dense(totalGenders))(wordSeqLSTM)
    activationForGender = Activation('softmax', name='activationForGender')(wordSeqTDForGender)
    wordSeqTDForNumber = TimeDistributed(Dense(totalNumbers))(wordSeqLSTM)
    activationForNumber = Activation('softmax', name='activationForNumber')(wordSeqTDForNumber)
    wordSeqTDForPerson = TimeDistributed(Dense(totalPersons))(wordSeqLSTM)
    activationForPerson = Activation('softmax', name='activationForPerson')(wordSeqTDForPerson)
    wordSeqTDForCase = TimeDistributed(Dense(totalCases))(wordSeqLSTM)
    activationForCase = Activation('softmax', name='activationForCase')(wordSeqTDForCase)
    wordSeqTDForVibh = TimeDistributed(Dense(totalVibhs))(wordSeqLSTM)
    activationForVibh = Activation('softmax', name='activationForVibh')(wordSeqTDForVibh)
    finalModel = Model(
        inputs=[charSeq, wordSeq], outputs=[activationForLcat, activationForGender, activationForNumber, activationForPerson, activationForCase, activationForVibh])
    finalModel.compile(optimizer='adam',
                       loss={'activationForLcat': 'sparse_categorical_crossentropy',
                             'activationForGender': 'sparse_categorical_crossentropy', 'activationForNumber': 'sparse_categorical_crossentropy',
                             'activationForPerson': 'sparse_categorical_crossentropy', 'activationForCase': 'sparse_categorical_crossentropy',
                             'activationForVibh': 'sparse_categorical_crossentropy'},
                       metrics=['accuracy'])
    print(finalModel.summary())
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    # batch_size = 16
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    finalModel.fit(trainGen, steps_per_epoch=steps, epochs=epochs, validation_data=valGen, validation_steps=valSteps, callbacks=[checkpointCallback])
    return finalModel


def main():
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--train', dest='tr',
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
    if not args.tr or not args.val or not args.mem or not args.w2i or not args.c2i or not args.wt:
        print(
            "Usage is python script_name --input input_file_name --w2v w2vecFile")
        exit(1)
    else:
        index2Lcat = loadObjectToPickleFile('index-to-lcat-tb.pkl')
        index2Gender = loadObjectToPickleFile('index-to-gender-tb.pkl')
        index2Number = loadObjectToPickleFile('index-to-number-tb.pkl')
        index2Person = loadObjectToPickleFile('index-to-person-tb.pkl')
        index2Case = loadObjectToPickleFile('index-to-case-tb.pkl')
        index2Vibh = loadObjectToPickleFile('index-to-vibh-tb.pkl')
        word2Index = loadObjectToPickleFile(args.w2i)
        char2Index = loadObjectToPickleFile(args.c2i)
        # print(char2Index)
        lcat2Index = createReverseIndex(index2Lcat)
        gender2Index = createReverseIndex(index2Gender)
        number2Index = createReverseIndex(index2Number)
        person2Index = createReverseIndex(index2Person)
        case2Index = createReverseIndex(index2Case)
        vibh2Index = createReverseIndex(index2Vibh)
        memoryMapped = np.memmap(args.mem, dtype='float32',
                                 mode='r', shape=(len(word2Index), 200))
        print(memoryMapped.shape)
        # epochs = 25
        epochs = 20
        batchSize = 16
        print('--TRAIN GEN--')
        trainLines = read_lines_from_file(args.tr)
        if trainLines[-1] != '\n':
            trainLines = trainLines + ['\n']
        totalSamples = len(findall('\n\n', ''.join(trainLines) + '\n', S))
        print('Total Samples', totalSamples)
        if totalSamples % batchSize == 0:
            steps = totalSamples // batchSize
        else:
            steps = totalSamples // batchSize + 1
        print(steps, 'STEPS')
        trainGen = createVectors(
            trainLines, word2Index, memoryMapped, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index)
        valLines = read_lines_from_file(args.val)
        if valLines[-1] != '\n':
            valLines = valLines + ['\n']
        valSamples = len(findall('\n\n', ''.join(valLines) + '\n', S))
        print('Total Val Samples', valSamples)
        if valSamples % batchSize == 0:
            valSteps = valSamples // batchSize
        else:
            valSteps = valSamples // batchSize + 1
        print(valSteps, 'VAL STEPS')
        valGen = createVectors(
            valLines, word2Index, memoryMapped, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index)
        del memoryMapped
        maxWordLength, maxSentenceLength = 18, 165
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        biLSTM = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, len(char2Index), trainGen, len(lcat2Index), len(gender2Index), len(number2Index), len(person2Index), len(case2Index), len(vibh2Index), args.wt, valGen, steps, valSteps, epochs=epochs)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    # tf.compat.v1.disable_eager_execution()
    config.gpu_options.allow_growth = True
#    sess = tf.compat.v1.Session(config=config)
#    sess.run(main)
    # tf.compat.v1.experimental.output_all_intermediates(True)
    main()
