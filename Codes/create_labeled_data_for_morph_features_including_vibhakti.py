"""Create labeled data for morph features."""
from argparse import ArgumentParser
from re import search
import os
from datetime import date


# Append the date with file names
todays_date = date.today()
todays_date_str = str(todays_date)
todays_date_str_split = todays_date_str.split('-')
todays_date_str_split_string = ''.join(todays_date_str_split[::-1])
# If same vibhakti/TAM marker appears in a sentence, then it is attached with 1/2/3 based on the number of occurrences
hindi_numbers = '[\U00000967-\U0000096F]+'
urdu_numbers = '[\U00000661-\U00000669]+'
telugu_numbers = '[\U00000C67-\U00000C6F]+'
kannada_numbers = '[\U00000CE7-\U00000CEF]+'
tamil_numbers = '[\U00000BE7-\U00000BEF]+'
malayalam_numbers = '[\U00000D67-\U00000D6F]+'
# For Marathi, Devnagari is used
english_numbers = '[\U00000031-\U00000039]+'
number_match_pattern = '|'.join([english_numbers, hindi_numbers, telugu_numbers, kannada_numbers, tamil_numbers, malayalam_numbers, urdu_numbers])


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def map_bis_to_lcat(bis_tag):
    """Convert BIS tag to lcat tag."""
    if search('N\_NN.*', bis_tag):
        return 'n'
    elif bis_tag == 'N_NST':
        return 'nst'
    elif search('^PR\_|^DM\_', bis_tag):
        return 'pn'
    elif search('^V\_', bis_tag):
        print('AA')
        return 'v'
    elif search('^RP\_|^CC\_', bis_tag):
        return 'avy'
    elif bis_tag == 'RB':
        return 'adv'
    elif bis_tag == 'JJ':
        return 'adj'
    elif bis_tag == 'PSP':
        return 'psp'
    elif bis_tag in ['RD_PUNC', 'RD_SYM']:
        return 'punc'
    elif bis_tag in ['RD_RDF', 'RD_UNK', 'RD_BUL']:
        return 'unk'
    elif bis_tag in ['QT_QTC', 'QT_QTO']:
        return 'num'
    elif bis_tag in ['QT_QTF', 'RD_ECH']:
        return 'avy'



def assign_default_bis_tag(pos):
    """Assign a default BIS tag if the tag is the top level tag."""
    if pos == 'N':
        return 'N_NN'
    elif pos == 'PR':
        return 'PR_PRP'
    elif pos == 'DM':
        return 'DM_DMD'
    elif pos == 'V':
        return 'V_VM'
    elif pos == 'CC':
        return 'CC_CCS'
    else:
        return pos


def process_lines_and_create_labeled_data_for_morph(lines, chunk_flag=1):
    """Process lines and create labeled data for morph fields."""
    lcat, gender, number, person, case, vibhakti = [''] * 6
    for index, line in enumerate(lines):
        all_feat = ''
        line = line.strip()
        if line:
            if not chunk_flag:
                token, pos, morph = line.split('\t')
            else:
                token, pos, chunk, morph = line.split('\t')
            pos = pos.replace('__', '_')
            if pos == 'QT_QTC' and ',' in token:
                morph = token.replace(',', '') + ',num,,,,,,'
            elif pos in ['RD_PUNC', 'RD_SYM']:
                if token != ',':
                    morph = token + ',punc,,,,,,'
                else:
                    morph = 'COMMA,punc,,,,,,'
            pos = assign_default_bis_tag(pos)
            fields = morph.split(',')
            print(fields, index)
            assert len(fields) == 8
            # if any of the fields are blank, the value is set to unk
            if not fields[1]:
                lcat = 'unk'
            else:
                lcat_mapped = map_bis_to_lcat(pos)
                # print(pos, lcat_mapped)
                if fields[1] == lcat_mapped:
                    lcat = fields[1]
                else:
                    lcat = lcat_mapped
            if not fields[2]:
                gender = 'unk'
                print('HE', gender)
            else:
                if fields[2] == 'ne':
                    fields[2] = 'n'
                gender = fields[2]
                print('SHE', gender)
            if not fields[3]:
                number = 'unk'
            else:
                number = fields[3]
            if not fields[4]:
                person = 'unk'
            else:
                person = fields[4]
            if not fields[5]:
                case = 'unk'
            else:
                case = fields[5]
            if not fields[6]:
                vibhakti = 'unk'
            elif fields[6] == '0':
                vibhakti = '0'
            else:
                number_match = search(number_match_pattern + '$', fields[6])
                if not number_match:
                    vibhakti = fields[6]
                else:
                    if number_match.start() > 0:
                        vibhakti = fields[6][: number_match.start()]
                    else:
                        vibhakti = fields[6][number_match.start() + 1:]
            print(token, pos, lcat, gender, number, person, case, vibhakti, chunk)
            if not chunk_flag:
                all_feat = '\t'.join([token, pos, lcat, gender, number, person, case, vibhakti])
            else:
                all_feat = '\t'.join([token, pos, lcat, gender, number, person, case, vibhakti, chunk])
            yield all_feat
        else:
            yield ''


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def write_generator_to_file(generator, file_path):
    """Write contents of generator into different files."""
    with open(file_path, 'a', encoding='utf-8') as all_feat_file:
        for all_feat in generator:
            if all_feat.strip():
                all_feat_file.write(all_feat + '\n')
            else:
                all_feat_file.write('\n')


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input folder/file path.')
    parser.add_argument('--lang', dest='lang', help='Enter the language.')
    parser.add_argument('--chunk', dest='chunk', help='Enter whether chunk annotation is present or not.', type=int, choices=[0, 1])
    args = parser.parse_args()
    if not os.path.isdir(args.inp):
        input_lines = read_lines_from_file(args.inp)
        all_features_generator = process_lines_and_create_labeled_data_for_morph(input_lines, args.chunk)
        if args.chunk:
            output_file_path = args.lang + '-token-pos-morph-with-vibh-chunk-' + todays_date_str_split_string + '.txt'
        else:
            output_file_path = args.lang + '-token-pos-morph-with-vibh-' + todays_date_str_split_string + '.txt'
        write_generator_to_file(all_features_generator, output_file_path)
    else:
        if args.chunk:
            output_folder = args.lang + '-token-pos-morph-with-vibh-chunk-' + todays_date_str_split_string + '.txt'
        else:
            output_folder = args.lang + '-token-pos-morph-with-vibh-' + todays_date_str_split_string + '.txt'
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        for root, dirs, files in os.walk(args.inp):
            for fl in files:
                print(fl)
                input_path = os.path.join(root, fl)
                input_lines = read_lines_from_file(input_path)
                all_features_generator = process_lines_and_create_labeled_data_for_morph(input_lines, args.chunk)
                file_types = ['dev', 'test', 'train']
                file_type_search = search('('+ '|'.join(file_types) + ')', fl)
                assert file_type_search is not None
                file_type = file_type_search.group(1)
                if args.chunk:
                    output_file_name = args.lang + '-' + file_type + '-token-pos-morph-with-vibh-chunk-' + todays_date_str_split_string + '-tab-separated.txt'
                else:
                    output_file_name = args.lang + '-' + file_type + '-token-pos-morph-with-vibh-' + todays_date_str_split_string + '-tab-separated.txt'
                output_path = os.path.join(output_folder, output_file_name)
                write_generator_to_file(all_features_generator, output_path)
    print(todays_date_str_split_string)
    

if __name__ == '__main__':
    main()
